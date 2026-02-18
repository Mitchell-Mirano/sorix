from sorix.tensor import Tensor, tensor

class NeuralNetwork:
    def __init__(self):
        super().__init__()
        self.device = None

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Debes implementar forward en la subclase.")

    def parameters(self):
        """Conserva tu implementación actual si te resulta útil para optimizadores,
        pero ya no dependemos de esto para mover a device."""
        params = []

        def _gather_params(obj):

            if hasattr(obj, "parameters") and callable(obj.parameters):
                params.extend(obj.parameters())

            elif isinstance(obj, NeuralNetwork):
                params.extend(obj.parameters())

            elif isinstance(obj, (list, tuple)):
                for layer in obj: _gather_params(layer)
                
            elif isinstance(obj, dict):
                for layer in obj.values(): _gather_params(layer)

        for layer in self.__dict__.values():
            _gather_params(layer)
        return params


    def to(self, device):
        """Mueve TODOS los tensores/capas/subredes al device."""

        def _apply(obj):
            # 1) Si el objeto tiene método .to, delegamos a él
            if hasattr(obj, "to") and callable(obj.to):
                return obj.to(device)

            # 2) Si es estructura compuesta → aplicar recursivamente
            if isinstance(obj, list):
                return [_apply(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_apply(v) for v in obj)
            if isinstance(obj, dict):
                return {k: _apply(v) for k, v in obj.items()}

            # 3) Otro tipo → devolver tal cual
            return obj

        # Aplicamos a todos los atributos del modelo
        for k, v in self.__dict__.items():
            setattr(self, k, _apply(v))

        self.device = device
        return self  # para encadenar llamadas
    

    def train(self):
        def _apply(obj):
            if hasattr(obj, "training"):
                obj.training = True
            if isinstance(obj, (list, tuple)):
                for o in obj: _apply(o)
            elif isinstance(obj, dict):
                for v in obj.values(): _apply(v)
        for v in self.__dict__.values():
            _apply(v)

    def eval(self):
        def _apply(obj):
            if hasattr(obj, "training"):
                obj.training = False
            if isinstance(obj, (list, tuple)):
                for o in obj: _apply(o)
            elif isinstance(obj, dict):
                for v in obj.values(): _apply(v)
        for v in self.__dict__.values():
            _apply(v)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def weights(self):

        if self.device == 'cpu':
            return self.__dict__
        return self.to('cpu').__dict__

    def load_weights(self, weights):
        self.__dict__.update(weights)
