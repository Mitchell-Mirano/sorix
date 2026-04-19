from __future__ import annotations
import numpy as np
from typing import Optional, Union, Tuple, Any
from sorix.tensor import Tensor, tensor, float32, is_grad_enabled
from sorix.cupy.cupy import _cupy_available
from sorix.nn.net import Module
from sorix.utils.conv import im2col_indices, col2im_indices

if _cupy_available:
    import cupy as cp
else:
    cp = None

class Conv2d(Module):
    """
    Applies a 2D convolution over an input signal composed of several input planes.
    
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
        device (str, optional): 'cpu' or 'cuda'.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
        device: str = 'cpu'
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
            
        self.device = device
        xp = cp if (device == 'cuda' and _cupy_available) else np
        
        # Xavier/Kaiming initialization
        std = xp.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        
        self.W = tensor(
            xp.random.normal(0, std, size=(out_channels, in_channels, *self.kernel_size)),
            device=device, requires_grad=True, dtype=float32
        )
        
        if bias:
            self.b = tensor(
                xp.zeros((out_channels, 1)),
                device=device, requires_grad=True, dtype=float32
            )
        else:
            self.b = None

    def __call__(self, X: Tensor) -> Tensor:
        N, C, H, W = X.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        
        # Calculate output dimensions
        out_h = (H + 2 * ph - kh) // sh + 1
        out_w = (W + 2 * pw - kw) // sw + 1
        
        xp = X.xp
        
        # im2col
        X_col = im2col_indices(X.data, kh, kw, self.padding, self.stride)
        W_col = self.W.data.reshape(self.out_channels, -1)
        
        out_data = W_col @ X_col
        if self.b is not None:
            out_data += self.b.data
            
        out_data = out_data.reshape(self.out_channels, out_h, out_w, N)
        out_data = out_data.transpose(3, 0, 1, 2)
        
        requires_grad = X.requires_grad or self.W.requires_grad or (self.b is not None and self.b.requires_grad)
        
        if not is_grad_enabled() or not requires_grad:
            return Tensor(out_data, device=self.device, requires_grad=False)
            
        deps = [X, self.W]
        if self.b is not None:
            deps.append(self.b)
            
        out = Tensor(out_data, deps, 'Conv2d', device=self.device, requires_grad=True)
        
        def _backward() -> None:
            if out.grad is None:
                return
            
            # Extract grad data and transpose to (OC, OH, OW, N) to match out_data before final transpose
            g_data = out.grad.data if isinstance(out.grad, Tensor) else out.grad
            g_data_reshaped = g_data.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
            
            # Gradient w.r.t bias
            if self.b is not None and self.b.requires_grad:
                self.b._accumulate_grad(xp.sum(g_data_reshaped, axis=1, keepdims=True))
                
            # Gradient w.r.t weights
            if self.W.requires_grad:
                dW = g_data_reshaped @ X_col.T
                self.W._accumulate_grad(dW.reshape(self.W.shape))
                
            # Gradient w.r.t input
            if X.requires_grad:
                dX_col = W_col.T @ g_data_reshaped
                dX = col2im_indices(dX_col, X.shape, kh, kw, self.padding, self.stride)
                X._accumulate_grad(dX)

        out._backward = _backward
        return out

class MaxPool2d(Module):
    """
    Applies a 2D max pooling over an input signal.
    """
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def __call__(self, X: Tensor) -> Tensor:
        N, C, H, W = X.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        
        out_h = (H + 2 * ph - kh) // sh + 1
        out_w = (W + 2 * pw - kw) // sw + 1
        
        xp = X.xp
        X_col = im2col_indices(X.data, kh, kw, self.padding, self.stride)
        X_col = X_col.reshape(C, kh * kw, -1)
        
        # Find max and argmax
        max_idx = xp.argmax(X_col, axis=1)
        out_data = xp.max(X_col, axis=1)
        
        out_data = out_data.reshape(C, out_h, out_w, N)
        out_data = out_data.transpose(3, 0, 1, 2)
        
        if not is_grad_enabled() or not X.requires_grad:
            return Tensor(out_data, device=X.device, requires_grad=False)
            
        out = Tensor(out_data, [X], 'MaxPool2d', device=X.device, requires_grad=True)
        
        def _backward() -> None:
            if out.grad is None:
                return
            
            g_data = out.grad.data if isinstance(out.grad, Tensor) else out.grad
            g_data = g_data.transpose(1, 2, 3, 0).reshape(C, -1)
            
            dX_col = xp.zeros_like(X_col)
            # Efficiently distribute gradients to max locations
            # We use advanced indexing to avoid loops
            n_cols = X_col.shape[2]
            c_idx = xp.arange(C)[:, xp.newaxis]
            col_idx = xp.arange(n_cols)
            
            dX_col[c_idx, max_idx, col_idx] = g_data
            
            dX_col = dX_col.reshape(C * kh * kw, -1)
            dX = col2im_indices(dX_col, X.shape, kh, kw, self.padding, self.stride)
            X._accumulate_grad(dX)

        out._backward = _backward
        return out

class Conv1d(Module):
    """
    Applies a 1D convolution over an input signal.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device: str = 'cpu'
    ) -> None:
        super().__init__()
        # 1D conv is just 2D conv with height 1
        self.conv2d = Conv2d(
            in_channels, out_channels, (1, kernel_size), 
            stride=(1, stride), padding=(0, padding), bias=bias, device=device
        )

    def __call__(self, X: Tensor) -> Tensor:
        # X: (N, C, L) -> Reshape to (N, C, 1, L)
        X_4d = X.unsqueeze(2)
        out_4d = self.conv2d(X_4d)
        return out_4d.squeeze(2)
    
    def parameters(self):
        return self.conv2d.parameters()

class MaxPool1d(Module):
    """
    Applies a 1D max pooling over an input signal.
    """
    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0
    ) -> None:
        super().__init__()
        self.pool2d = MaxPool2d((1, kernel_size), stride=(1, stride) if stride else None, padding=(0, padding))

    def __call__(self, X: Tensor) -> Tensor:
        X_4d = X.unsqueeze(2)
        out_4d = self.pool2d(X_4d)
        return out_4d.squeeze(2)
