from __future__ import annotations
from typing import Union, List, Tuple, Optional, Any
from .tensor import Tensor
import numpy as np
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp

def grad(
    outputs: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
    inputs: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
    grad_outputs: Optional[Union[Tensor, List[Tensor], Tuple[Tensor, ...], Any]] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    allow_unused: bool = False
) -> Tuple[Tensor, ...]:
    """
    Computes and returns the sum of gradients of outputs w.r.t. the inputs.
    
    Args:
        outputs: Tensors of which the gradient is to be computed.
        inputs: Tensors w.r.t which the gradient will be computed.
        grad_outputs: The "vector" in the vector-Jacobian product. 
            Should be the same size as outputs.
        retain_graph: If False, the graph used to compute the grads will be freed.
        create_graph: If True, graph of the derivative will be constructed, 
            allowing to compute higher order derivative products.
        allow_unused: If False, raises an error if an input is not part of the graph.
            
    Returns:
        A tuple of Tensors containing the gradients for each input.
    """
    from .tensor import set_grad_enabled, is_grad_enabled

    # 1. Standardize and validate inputs
    outputs = [outputs] if isinstance(outputs, Tensor) else list(outputs)
    inputs = [inputs] if isinstance(inputs, Tensor) else list(inputs)
    
    for inp in inputs:
        if not inp.requires_grad:
            raise RuntimeError("One of the inputs does not require grad. Set requires_grad=True for all inputs.")

    if grad_outputs is None:
        grad_outputs = [None] * len(outputs)
    else:
        if isinstance(grad_outputs, (Tensor, np.ndarray)) or (_cupy_available and isinstance(grad_outputs, cp.ndarray)):
            grad_outputs = [grad_outputs]
        grad_outputs = list(grad_outputs)
        
    if len(grad_outputs) != len(outputs):
        raise ValueError("grad_outputs must be the same length as outputs")

    # 2. Build topological order — always save/restore _prev so nothing is freed.
    topo: List[Tensor] = []
    visited = set()
    
    def build_topo(t: Tensor) -> None:
        if id(t) not in visited:
            visited.add(id(t))
            for child in t._prev:
                build_topo(child)
            topo.append(t)
            
    for out in outputs:
        build_topo(out)
        
    # 3. Save ALL existing gradients and graph edges (strict isolation)
    saved_grads = {id(node): node.grad for node in topo}
    saved_prev  = {id(node): node._prev for node in topo}
    for node in topo:
        node.grad = None

    if retain_graph is None:
        retain_graph = create_graph

    prev_grad_enabled = is_grad_enabled()
    set_grad_enabled(create_graph)
    
    try:
        # 4. Set seed gradients
        for out, go in zip(outputs, grad_outputs):
            if go is None:
                if out.data.size != 1:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs.")
                xp = cp if out.device.type == 'cuda' else np
                seed_data = xp.ones_like(out.data, dtype=out.data.dtype)
                if create_graph:
                    out.grad = Tensor(seed_data, device=out.device, requires_grad=True)
                else:
                    out.grad = seed_data
            else:
                out.grad = go if isinstance(go, Tensor) else Tensor(go, device=out.device, requires_grad=create_graph)

        # 5. Execute backward pass — restore _prev first so _backward() works correctly
        for node in topo:
            node._prev = saved_prev[id(node)]

        for node in reversed(topo):
            if node.grad is not None:
                node._backward()
    finally:
        set_grad_enabled(prev_grad_enabled)

    # 6. Collect results before restoring
    results = []
    for inp in inputs:
        grad_val = inp.grad

        if grad_val is None:
            if not allow_unused:
                raise RuntimeError(
                    "One of the inputs was not used in the graph. "
                    "Set allow_unused=True if this is expected."
                )
            results.append(None)
            continue

        # Always return a Tensor so callers can use .numpy(), .item(), etc.
        if isinstance(grad_val, Tensor):
            results.append(grad_val)
        else:
            results.append(Tensor(grad_val.copy(), device=inp.device))

    # 7. Restore ALL original gradients and graph structure
    for node in topo:
        node.grad  = saved_grads[id(node)]
        node._prev = saved_prev[id(node)]

    # 8. If caller EXPLICITLY requested graph release, clear _prev now
    # Default is retain_graph=True to allow multiple calls on the same graph
    if not retain_graph:
        for node in topo:
            node._prev = set()
        
    return tuple(results)
