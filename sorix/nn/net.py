from __future__ import annotations
from typing import List, Dict, Any, Iterator, Optional, Union, Set
from sorix.tensor import Tensor, tensor

def _add_indent(s: str, num_spaces: int) -> str:
    s = s.split('\n')
    if len(s) == 1:
        return s[0]
    first = s.pop(0)
    s = [(num_spaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class Module:
    """
    Base class for all neural network modules.
    
    Your models should also subclass this class.
    """
    def __init__(self) -> None:
        super().__init__()
        self.device: str = 'cpu'
        self.training: bool = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError("Debes implementar forward en la subclase.")

    def parameters(self) -> List[Tensor]:
        """
        Returns an iterator over module parameters (tensors that require gradients).
        """
        params: List[Tensor] = []
        visited: Set[int] = set()

        def _gather_params(obj: Any) -> None:
            if id(obj) in visited:
                return
            visited.add(id(obj))

            if isinstance(obj, Tensor):
                if obj.requires_grad:
                    params.append(obj)
            elif hasattr(obj, "parameters") and callable(obj.parameters) and obj is not self:
                # If the object has its own parameters() method, use it
                params.extend(obj.parameters())
            elif hasattr(obj, "__dict__"):
                # Recurse into attributes
                for k, v in obj.__dict__.items():
                    if not k.startswith('_'):
                        _gather_params(v)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _gather_params(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    _gather_params(item)

        _gather_params(self)
        return params

    def to(self, device: str) -> Module:
        """
        Moves all model parameters and buffers to the specified device.
        
        Args:
            device: 'cpu' or 'gpu'.
        """
        self.device = device

        def _apply(obj: Any) -> Any:
            if hasattr(obj, "to") and callable(obj.to) and obj is not self:
                return obj.to(device)
            
            if isinstance(obj, Tensor):
                return obj.to(device)

            if isinstance(obj, list):
                return [_apply(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_apply(v) for v in obj)
            if isinstance(obj, dict):
                return {k: _apply(v) for k, v in obj.items()}
            
            if hasattr(obj, "__dict__") and obj is not self:
                 # Try to move attributes of non-Module objects
                 for k, v in obj.__dict__.items():
                     if not k.startswith('_'):
                         setattr(obj, k, _apply(v))
            
            return obj

        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                setattr(self, k, _apply(v))

        return self

    def train(self) -> None:
        """Sets the module in training mode."""
        self.training = True
        def _apply(obj: Any) -> None:
            if hasattr(obj, "training"):
                obj.training = True
            if isinstance(obj, (list, tuple)):
                for o in obj: _apply(o)
            elif isinstance(obj, dict):
                for v in obj.values(): _apply(v)
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                _apply(v)

    def eval(self) -> None:
        """Sets the module in evaluation mode."""
        self.training = False
        def _apply(obj: Any) -> None:
            if hasattr(obj, "training"):
                obj.training = False
            if isinstance(obj, (list, tuple)):
                for o in obj: _apply(o)
            elif isinstance(obj, dict):
                for v in obj.values(): _apply(v)
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                _apply(v)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def state_dict(self) -> Dict[str, Tensor]:
        """
        Returns a dictionary containing the whole state of the module (parameters and buffers).
        """
        state: Dict[str, Tensor] = {}

        def _get_state(obj: Any, prefix: str) -> None:
            for name, val in obj.__dict__.items():
                if name.startswith('_') or name == 'device' or name == 'training':
                    continue
                
                key = prefix + name
                if isinstance(val, Tensor):
                    state[key] = val
                elif hasattr(val, 'state_dict') and callable(val.state_dict) and val is not self:
                    sub_state = val.state_dict()
                    for sk, sv in sub_state.items():
                        state[key + '.' + sk] = sv
                elif hasattr(val, '__dict__'):
                     for kn, kv in val.__dict__.items():
                         if isinstance(kv, Tensor):
                             state[key + '.' + kn] = kv
            
        _get_state(self, "")
        return state

    def extra_repr(self) -> str:
        """
        Set the extra representation of the module.
        Should be overridden by subclasses.
        """
        return ""

    def __repr__(self) -> str:
        # Get children modules
        child_modules = {}
        # We need to consider both named attributes and _modules for Sequential
        if hasattr(self, '_modules'):
            child_modules = self._modules
        else:
            for name, module in self.__dict__.items():
                if isinstance(module, Module) and module is not self:
                    child_modules[name] = module

        main_str = self.__class__.__name__ + '('
        if child_modules:
            main_str += '\n'
            for name, module in child_modules.items():
                mod_str = repr(module)
                mod_str = _add_indent(mod_str, 2)
                main_str += f'  ({name}): {mod_str}\n'
            main_str += ')'
        else:
            extra_lines = []
            extra_repr = self.extra_repr()
            if extra_repr:
                extra_lines = extra_repr.split('\n')
            if len(extra_lines) > 1:
                main_str += '\n  ' + '\n  '.join(extra_lines) + '\n)'
            elif len(extra_lines) == 1:
                main_str += extra_lines[0] + ')'
            else:
                main_str += ')'
        return main_str

    def __str__(self) -> str:
        return self.__repr__()

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """
        Copies parameters and buffers from state_dict into this module and its descendants.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, Tensor):
                    own_state[name].data = param.data
                    own_state[name].to(own_state[name].device) # Ensure device consistency
                else:
                    pass
            else:
                pass


class Sequential(Module):
    """
    A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    
    Examples:
        ```python
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        ```
    """
    def __init__(self, *args: Any) -> None:
        super().__init__()
        self._modules: Dict[str, Module] = {}
        if len(args) == 1 and isinstance(args[0], dict):
            for name, module in args[0].items():
                setattr(self, name, module)
            self._modules = args[0]
        else:
            for idx, module in enumerate(args):
                name = str(idx)
                setattr(self, name, module)
                self._modules[name] = module

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x

    def __getitem__(self, idx: Union[int, slice, str]) -> Union[Module, Sequential]:
        if isinstance(idx, slice):
            return Sequential(dict(list(self._modules.items())[idx]))
        if isinstance(idx, int):
            if idx < 0: idx += len(self._modules)
            return self._modules[str(idx)]
        return self._modules[str(idx)]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())


