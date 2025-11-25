# ⚡ El Objeto Tensor

El **Tensor** es la estructura de datos central de Sorix, análoga a los arrays de NumPy, pero con la capacidad crucial de registrar el historial de operaciones para la diferenciación automática (Autograd).

## Creación Básica

Se puede inicializar un tensor a partir de cualquier array de NumPy o lista de Python.

!!! tip "Tip"
    Si estableces `requires_grad=True`, Sorix comenzará a rastrear las operaciones para calcular los gradientes más tarde.

```python
import numpy as np
from sorix import Tensor

# Crear un tensor a partir de un array de NumPy
data = np.array([[1.0, 2.0], [3.0, 4.0]])
x = Tensor(data, requires_grad=True)

print("El tensor:", x)
print("Su dimensión:", x.shape)
print("Requiere gradiente:", x.requires_grad)
```