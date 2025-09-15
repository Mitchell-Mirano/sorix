from allison.tensor.tensor import tensor
from allison.cupy.cupy import _cupy_available
import numpy as np

if _cupy_available:
    import cupy as cp
    

class Linear:
    def __init__(self, features: int, neurons: int,bias=True, init='he',device='cpu'):

        if device == 'gpu' and not _cupy_available:
            raise Exception('Cupy is not available')
        
        self.device = device

        xp = cp if device == 'gpu' else np
        
        if init not in ['he', 'xavier']:
            raise ValueError(f'Invalid initialization method: {init}. Valid methods are "he" and "xavier"')
        
        if init == 'he':
            self.std_dev = xp.sqrt(2.0 / features)  # He init para ReLU
        elif init == 'xavier':
            self.std_dev = xp.sqrt(2.0 / (features + neurons))  # Xavier init para tanh

        self.bias = bias
        self.W = tensor(xp.random.normal(0, self.std_dev, size=(features, neurons)),device=self.device,requires_grad=True)
        self.b = tensor(xp.zeros((1, neurons)),device=self.device,requires_grad=True)  if self.bias else None

    def __call__(self, X: tensor):
        if self.bias:
            return X @ self.W + self.b  
        return X @ self.W
    
    def to(self, device):

        if device == self.device:
            return self
        
        self.W = self.W.to(device)

        if self.bias:
            self.b = self.b.to(device)
        self.device = device
        return self
    
    def parameters(self):
        if self.bias:
            return [self.W, self.b] 
        return [self.W]
    
    @property
    def coef_(self):
        return self.W.data.flatten()
        
    @property
    def intercept_(self):
        return self.b.item()


class Relu:
    def __call__(self, X: tensor):

        xp = cp if X.device == 'gpu' else np

        out = tensor(xp.maximum(0, X.data), (X,), 'ReLU',device=X.device,requires_grad=X.requires_grad)
        
        def _backward():
            # Usar other.data para claridad
            X.grad += out.grad * (X.data > 0)
        out._backward = _backward
        return out


class Sigmoid:
    def __call__(self, X: tensor):

        xp = cp if X.device == 'gpu' else np

        out = tensor(1 / (1 + xp.exp(-X.data)), (X,), 'Sigmoid',device=X.device,requires_grad=X.requires_grad)
        
        def _backward():
            # Usar other.data para claridad
            X.grad += out.grad * out.data * (1 - out.data)
        out._backward = _backward
        return out
    
    
class Tanh:
    def __call__(self, X: tensor):

        xp = cp if X.device == 'gpu' else np

        out = tensor(xp.tanh(X.data), (X,), 'Tanh',device=X.device,requires_grad=X.requires_grad)
        
        def _backward():
            # Usar other.data para claridad
            X.grad += out.grad * (1 - out.data**2)
        out._backward = _backward
        return out

class BatchNorm1D:
    def __init__(self, features: int, alpha: float = 0.9, epsilon: float = 1e-5, device='cpu'):
        self.gamma = tensor(np.ones((1, features)), requires_grad=True)
        self.beta = tensor(np.zeros((1, features)), requires_grad=True)

        # buffers (no requieren gradiente)
        self.running_mean = np.zeros((1, features), dtype=np.float32)
        self.running_var = np.ones((1, features), dtype=np.float32)

        self.alpha = alpha
        self.epsilon = epsilon
        self.device = device
        self.training = True

    def __call__(self, X: tensor):
        xp = cp if X.device == 'gpu' else np

        if self.training:
            # estadísticas del batch
            batch_mean = xp.mean(X.data, axis=0, keepdims=True)
            batch_var = xp.var(X.data, axis=0, keepdims=True)

            # actualizar los buffers (NO tensores, solo numpy/cupy arrays)
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * batch_mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * batch_var

            mean = batch_mean
            var = batch_var
        else:
            # usar estadísticas acumuladas
            mean = self.running_mean
            var = self.running_var

        # normalizar
        X_norm = (X - mean) / xp.sqrt(var + self.epsilon)
        out = self.gamma * X_norm + self.beta
        return out

    def to(self, device):
        if device == self.device:
            return self

        if device == 'gpu' and not _cupy_available:
            raise Exception('Cupy is not available')

        if device == 'gpu':
            self.running_mean = cp.array(self.running_mean)
            self.running_var = cp.array(self.running_var)
        else:  # cpu
            self.running_mean = cp.asnumpy(self.running_mean)
            self.running_var = cp.asnumpy(self.running_var)

        self.gamma = self.gamma.to(device)
        self.beta = self.beta.to(device)
        self.device = device
        return self

    def parameters(self):
        return [self.gamma, self.beta]
    


class Conv2d:
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3,3),
                 bias=False, mode='valid', stride=(1,1), dilation=(1,1),
                 init='he', device='cpu', dtype='float32'):
        if device == 'gpu' and not _cupy_available:
            raise RuntimeError("Cupy no disponible para device='gpu'")

        self.device = device
        self.xp = cp if device == 'gpu' else np
        xp = self.xp

        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        self.kernel_size = (kH, kW)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias_flag = bias
        self.mode = mode
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.dtype = dtype

        fan_in = in_channels * kH * kW
        fan_out = out_channels * kH * kW
        if init == 'he':
            std = (2.0 / fan_in) ** 0.5
        elif init == 'xavier':
            std = (2.0 / (fan_in + fan_out)) ** 0.5
        else:
            raise ValueError("init must be 'he' or 'xavier'")

        W_np = (xp.random.randn(out_channels, in_channels, kH, kW) * std).astype(self.dtype)
        self.W = tensor(xp.ascontiguousarray(W_np), device=device, requires_grad=True)

        if self.bias_flag:
            b_np = xp.zeros((out_channels,), dtype=self.dtype)
            self.b = tensor(b_np, device=device, requires_grad=True)
        else:
            self.b = None

    # ---------- util: padding calc ----------
    def _get_padding(self, H, W):
        kH, kW = self.kernel_size
        dil_h, dil_w = self.dilation
        effective_kh = (kH - 1) * dil_h + 1
        effective_kw = (kW - 1) * dil_w + 1

        if self.mode == 'valid':
            pad_h = pad_w = 0
        elif self.mode == 'same':
            # symmetric padding (floor/ceil choice: here floor both sides)
            pad_h = (effective_kh - 1) // 2
            pad_w = (effective_kw - 1) // 2
        else:
            raise ValueError("mode must be 'valid' or 'same'")
        return pad_h, pad_w

    # ---------- util: im2col (works for numpy and cupy) ----------
    def _im2col(self, x):
        """
        x: array shape (N, C, H, W)
        returns cols shape (N, K, L) where K=C*kH*kW and L=H_out*W_out
        and H_out, W_out
        """
        xp = self.xp
        N, C, H, W = x.shape
        kH, kW = self.kernel_size
        stride_h, stride_w = self.stride
        dil_h, dil_w = self.dilation

        pad_h, pad_w = self._get_padding(H, W)
        if pad_h > 0 or pad_w > 0:
            x_padded = xp.pad(x, ((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)))
        else:
            x_padded = x

        H_p, W_p = x_padded.shape[2], x_padded.shape[3]
        eff_kh = (kH - 1) * dil_h + 1
        eff_kw = (kW - 1) * dil_w + 1

        H_out = (H_p - eff_kh) // stride_h + 1
        W_out = (W_p - eff_kw) // stride_w + 1

        K = C * kH * kW
        L = H_out * W_out
        cols = xp.empty((N, K, L), dtype=x_padded.dtype)

        col_idx = 0
        for i in range(kH):
            i_off = i * dil_h
            for j in range(kW):
                j_off = j * dil_w
                patch = x_padded[:, :, i_off:i_off + stride_h*H_out:stride_h,
                                        j_off:j_off + stride_w*W_out:stride_w]  # (N,C,H_out,W_out)
                cols[:, col_idx*C:(col_idx+1)*C, :] = patch.reshape(N, C, -1)
                col_idx += 1

        return cols, H_out, W_out, pad_h, pad_w

    def _col2im(self, cols, x_shape, H_out, W_out, pad_h, pad_w):
        xp = self.xp
        N, C, H, W = x_shape
        kH, kW = self.kernel_size
        stride_h, stride_w = self.stride
        dil_h, dil_w = self.dilation

        H_p = H + 2*pad_h
        W_p = W + 2*pad_w
        x_padded = xp.zeros((N, C, H_p, W_p), dtype=cols.dtype)

        col_idx = 0
        for i in range(kH):
            i_off = i * dil_h
            for j in range(kW):
                j_off = j * dil_w
                patch = cols[:, col_idx*C:(col_idx+1)*C, :].reshape(N, C, H_out, W_out)
                x_padded[:, :, i_off:i_off + stride_h*H_out:stride_h,
                             j_off:j_off + stride_w*W_out:stride_w] += patch
                col_idx += 1

        if pad_h == 0 and pad_w == 0:
            return x_padded
        return x_padded[:, :, pad_h:pad_h + H, pad_w:pad_w + W]

    # ---------- forward ----------
    def __call__(self, other):
        # ensure tensor & device match
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        if other.device != self.device:
            other = other.to(self.device)

        xp = self.xp

        N, C, H, W = other.data.shape
        assert C == self.in_channels, f"Esperado {self.in_channels}, recibido {C}"

        # im2col
        cols, H_out, W_out, pad_h, pad_w = self._im2col(other.data)  # cols: (N, K, L)
        K = cols.shape[1]
        L = cols.shape[2]

        # W_col: (out_channels, K)
        W_col = self.W.data.reshape(self.out_channels, -1)

        # Batch GEMM: cols.T matmul W_col^T
        # cols_transposed: (N, L, K)
        cols_t = cols.transpose(0, 2, 1)
        out_t = xp.matmul(cols_t, W_col.T)  # shape (N, L, out_channels)
        out_data = out_t.transpose(0, 2, 1).reshape(N, self.out_channels, H_out, W_out)

        if self.bias_flag:
            out_data = out_data + self.b.data.reshape(1, -1, 1, 1)

        out = tensor(
            xp.ascontiguousarray(out_data.astype(self.dtype)),
            [other],
            'Conv2d',
            device=self.device,
            requires_grad=(other.requires_grad or self.W.requires_grad or (self.b is not None and self.b.requires_grad))
        )

        # capture needed values for backward
        saved = {
            'cols': cols, 'x_shape': other.data.shape,
            'H_out': H_out, 'W_out': W_out,
            'pad_h': pad_h, 'pad_w': pad_w,
            'W_col_shape': W_col.shape
        }

        def _backward():
            dY = out.grad  # (N, out_channels, H_out, W_out)
            if dY is None:
                return

            # ensure grads initialized
            if other.requires_grad and (other.grad is None):
                other.grad = xp.zeros_like(other.data)
            if self.W.requires_grad and (self.W.grad is None):
                self.W.grad = xp.zeros_like(self.W.data)
            if self.bias_flag and (self.b.grad is None):
                self.b.grad = xp.zeros_like(self.b.data)

            N_local = dY.shape[0]
            L_local = self.saved_L = saved['H_out'] * saved['W_out']

            # reshape dY -> (N, out_channels, L)
            dY_cols = dY.reshape(N_local, self.out_channels, -1)  # (N, O, L)

            cols_local = saved['cols']  # (N, K, L)

            # dW: sum_n dY_cols[n] @ cols[n].T -> use einsum
            if self.W.requires_grad:
                dW_col = xp.einsum('nol,nkl->ok', dY_cols, cols_local)  # (O, K)
                self.W.grad += dW_col.reshape(self.W.data.shape)

            # db:
            if self.bias_flag and self.b.requires_grad:
                db = dY.sum(axis=(0, 2, 3))  # (out_channels,)
                self.b.grad += db

            # dX: for each n, dX_cols[n] = W_col.T @ dY_cols[n]
            # vectorized:
            dX_cols = xp.einsum('ok,nol->nkl', W_col, dY_cols)  # (N, K, L)

            # col2im -> dX
            dX = self._col2im(dX_cols, saved['x_shape'], saved['H_out'], saved['W_out'], saved['pad_h'], saved['pad_w'])
            if other.requires_grad:
                other.grad += dX

        out._backward = _backward
        return out

    # ---------- convenience ----------
    def to(self, device):
        if device == self.device:
            return self
        if device == 'gpu' and not _cupy_available:
            raise RuntimeError('Cupy not available')
        self.W = self.W.to(device)
        if self.b is not None:
            self.b = self.b.to(device)
        self.device = device
        self.xp = cp if device == 'gpu' else np
        return self

    def parameters(self):
        return [self.W, self.b] if self.bias_flag else [self.W]
    

class MaxPool2d:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, other: tensor):
        xp = cp if other.device == "gpu" else np
        N, C, H, W = other.data.shape
        k = self.kernel_size
        s = self.stride

        # calcular salida
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1

        # im2col: (N, C, k*k, out_h*out_w)
        cols = xp.lib.stride_tricks.sliding_window_view(
            other.data, (k, k), axis=(2, 3)
        )  # (N, C, H-k+1, W-k+1, k, k)

        # aplicar stride
        cols = cols[:, :, ::s, ::s, :, :]  # (N,C,out_h,out_w,k,k)

        # a plano para argmax
        cols_reshaped = cols.reshape(N, C, out_h, out_w, -1)  # (N,C,out_h,out_w,k*k)

        # máximo y sus índices
        out_data = cols_reshaped.max(axis=-1)  # (N,C,out_h,out_w)
        self.argmax = cols_reshaped.argmax(axis=-1)  # (N,C,out_h,out_w)

        out = tensor(
            xp.ascontiguousarray(out_data),
            [other],
            "MaxPool2d",
            device=other.device,
            requires_grad=other.requires_grad,
        )

        def _backward():
            if other.requires_grad:
                grad = xp.zeros_like(other.data)

                # expandimos gradiente
                dY = out.grad.reshape(N, C, out_h, out_w, 1)  # (N,C,out_h,out_w,1)

                # construir máscara one-hot según argmax
                mask = xp.zeros_like(cols_reshaped, dtype=bool)  # (N,C,out_h,out_w,k*k)
                mask[xp.arange(N)[:, None, None, None],
                     xp.arange(C)[None, :, None, None],
                     xp.arange(out_h)[None, None, :, None],
                     xp.arange(out_w)[None, None, None, :],
                     self.argmax] = True

                # distribuir gradiente en la ventana correspondiente
                dcols = mask * dY  # (N,C,out_h,out_w,k*k)

                # devolver a (N,C,out_h,out_w,k,k)
                dcols = dcols.reshape(N, C, out_h, out_w, k, k)

                # sumamos en la posición correcta
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, w_start = i * s, j * s
                        grad[:, :, h_start:h_start+k, w_start:w_start+k] += dcols[:, :, i, j]

                other.grad += grad

        out._backward = _backward
        return out
