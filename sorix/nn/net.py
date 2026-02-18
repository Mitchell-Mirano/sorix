from sorix.tensor import Tensor, tensor

class Module:
    def __init__(self):
        super().__init__()
        self.device = 'cpu'
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Debes implementar forward en la subclase.")

    def parameters(self):
        """
        Returns an iterator over module parameters.
        """
        params = []
        visited = set()

        def _gather_params(obj):
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

    def to(self, device):
        """Mueve TODOS los tensores/capas/subredes al device."""
        self.device = device

        def _apply(obj):
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

    def train(self):
        self.training = True
        def _apply(obj):
            if hasattr(obj, "training"):
                obj.training = True
            if isinstance(obj, (list, tuple)):
                for o in obj: _apply(o)
            elif isinstance(obj, dict):
                for v in obj.values(): _apply(v)
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                _apply(v)

    def eval(self):
        self.training = False
        def _apply(obj):
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
    
    def state_dict(self):
        """
        Returns a dictionary containing a whole state of the module.
        """
        state = {}

        def _get_state(obj, prefix):
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

    def load_state_dict(self, state_dict):
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

    def weights(self):
        """Deprecated: use state_dict() instead."""
        return self.state_dict()

    def load_weights(self, weights):
        """Deprecated: use load_state_dict() instead."""
        self.load_state_dict(weights)


