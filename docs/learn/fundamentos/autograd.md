## 🧠 Autograd: Cálculo Automático de Derivadas

El motor `autograd` de Sorix te permite calcular las derivadas de las funciones de manera automática, lo que te permite enfocarte en la definición de la función de pérdida y no preocuparte por la implementación de las derivadas.

En esencia, cada operación que realizas con un **Tensor** crea un **gráfico de cómputo** (Computation Graph) que registra cómo se llegó al valor final. El método `.backward()` recorre este gráfico a la inversa (de la pérdida a los parámetros) aplicando la **regla de la cadena** para calcular los gradientes.

### El Ejemplo Clásico: Error Cuadrático Medio (MSE)

El Error Cuadrático Medio (MSE) es la función de pérdida más común para los problemas de regresión. En un modelo lineal simple ($\hat{y} = xw$), el objetivo es calcular la derivada de la pérdida ($L$) con respecto al parámetro $w$: $\frac{\partial L}{\partial w}$.

Aquí se demuestra cómo realizar este cálculo con Sorix:

```python
# 1. Importar Tensor (asumimos que ya está disponible)
from sorix import Tensor
import numpy as np

# --- 1. Datos de Entrada ---
# Característica de entrada X
X = Tensor(np.array([1.0, 2.0]))
# Etiqueta real Y
Y_true = np.array([3.0, 4.0])

# --- 2. Parámetro a Optimizar ---

# El tensor 'w' es el parámetro del modelo. 
# requires_grad=True es CRUCIAL para que autograd rastree sus operaciones.
w = Tensor(1.0, requires_grad=True)

# --- 3. Pase Adelante (Forward Pass) ---

# Modelo lineal simple: Y_hat = X * w
Y_hat = X * w

# --- 4. Cálculo de la Pérdida (MSE) ---

# Error = (Y_hat - Y_true)
error = Y_hat - Y_true

# Pérdida (L) = mean(error**2)
loss = (error**2).mean()

# --- 5. Backpropagation y Gradiente ---

# Ejecutar el backpropagation. Esto computa los gradientes de 'loss' 
# con respecto a todos los tensores que tengan requires_grad=True.
loss.backward()

# --- 6. Resultado ---

print(f"Valor de la pérdida (L): {loss.item():.4f}")
print(f"Gradiente de W (dL/dw): {w.grad:.4f}")
```

### Explicación del Resultado

Cuando se llama a `loss.backward()`, el gráfico de cómputo se evalúa. El valor final del gradiente se almacena en el atributo **`.grad`** del tensor `w`.

En este ejemplo:

  * Si $w=1.0$, la predicción es $\hat{y} = [1.0, 2.0]$.
  * El error es $Y_{true} - \hat{y} = [3, 4] - [1, 2] = [2, 2]$.
  * El gradiente $\frac{\partial L}{\partial w}$ calculado por `autograd` es **-6.0** (verificado manualmente por la regla de la cadena).

El motor `autograd` ha calculado el valor exacto que se usaría para actualizar el peso `w` en un algoritmo de optimización como el Descenso de Gradiente.