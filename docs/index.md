
# ⚡ Sorix: Aprendizaje Profundo Minimalista y de Alto Rendimiento

**Sorix** es una librería de Machine Learning y Deep Learning diseñada para ser **minimalista y de alto rendimiento**. Su principal característica es la capacidad de ejecutar redes neuronales directamente sobre **NumPy** con un uso mínimo de recursos.

Inspirada en la **API de PyTorch**, Sorix mantiene una interfaz clara e intuitiva que permite una adopción rápida, sin comprometer la eficiencia. Su arquitectura facilita una transición fluida desde el prototipo de investigación hasta la producción, sin necesidad de reescritura estructural.



---

## ✨ Características Distintivas

Aprovecha la sintaxis expresiva y familiar de Sorix, construida para ser ligera y potente:

* **Núcleo de Cálculo sobre NumPy/CuPy:**
    * Ejecuta redes neuronales optimizadas sobre **NumPy** (CPU) con aceleración **GPU opcional** a través de **CuPy**.
* **Diseño Ligero y Eficiente:**
    * Ideal para entornos con **recursos computacionales limitados** o cuando se requiere una baja sobrecarga.
* **API Familiar y Clara:**
    * Basada en los principios de diseño de **PyTorch**, lo que garantiza una curva de aprendizaje corta para usuarios familiarizados con otros *frameworks*.
* **Ruta Directa a Producción:**
    * Desarrolla modelos listos para producción sin la necesidad de reescribir o migrar a otros *frameworks* pesados.

> Sorix equilibra simplicidad, rendimiento y escalabilidad, permitiendo la comprensión completa de la mecánica interna de los modelos mientras se construyen soluciones listas para el despliegue en el mundo real.

---

## 📦 Instalación

Puedes instalar Sorix fácilmente usando tus herramientas favoritas de gestión de paquetes de Python.

=== "pip"

    Instala Sorix desde PyPI:
    ```bash
    pip install sorix
    ```

=== "Poetry"

    Añade Sorix a tu proyecto con Poetry:
    ```bash
    poetry add sorix
    ```

=== "uv"

    Usa el gestor de paquetes UV (de Astral):
    ```bash
    uv add sorix
    ```

---

## 🚀 Inicio Rápido: Primeros Pasos

A continuación, se muestran ejemplos que ilustran el sistema de diferenciación automática (`autograd`) y el uso de módulos de red neuronal (`nn`).

### Autograd: Cálculo Automático de Derivadas

El motor `autograd` de Sorix te permite calcular las derivadas de las funciones:

```python
from sorix import tensor

# 1. Crear tensores y habilitar el rastreo de gradientes
x = tensor([2.0], requires_grad=True)
w = tensor([3.0], requires_grad=True)
b = tensor([1.0], requires_grad=True)

# 2. Definir una función simple: y = w*x + b
y = w * x + b

# 3. Calcular gradientes mediante retropropagación
y.backward()

# Resultado
print("dy/dx:", x.grad)   # → 3.0
print("dy/dw:", w.grad)   # → 2.0
print("dy/db:", b.grad)   # → 1.0
```

### Regresión Lineal con `nn` y `optim`

Un ejemplo completo de entrenamiento con capas, pérdida y optimizador:

```python
import numpy as np
from sorix import tensor
from sorix.nn import Linear, MSELoss
from sorix.optim import SGD

# Generación de datos sintéticos (y = 3x + 2 + ruido)
X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = 3 * X + 2 + 0.1 * np.random.randn(*X.shape)

# Conversión a tensores de Sorix (dispositivo: CPU)
X_tensor = tensor(X, device="cpu")
y_tensor = tensor(y, device="cpu")

# Definición del modelo y entrenamiento
features, outputs = 1, 1
model = Linear(features, outputs)
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.1)

# Bucle de entrenamiento (200 épocas)
for epoch in range(200):
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/200] - Loss: {loss.item():.4f}")

# Parámetros aprendidos
print("Peso aprendido:", model.coef_)
print("Sesgo aprendido:", model.intercept_)
```

-----

## 📖 Documentación y Ejemplos Interactivos

Explora la funcionalidad completa de Sorix con nuestros *notebooks* interactivos.

| Nombre del Ejemplo | Descripción | Enlace |
| :--- | :--- | :--- |
| **Tensor Basics** | Creación y manipulación fundamental de tensores. | [Ver Notebook ➡️](https://github.com/Mitchell-Mirano/sorix/blob/develop/examples/basics/1-tensor.ipynb) |
| **Regresión** | Implementación de un modelo de regresión simple. | [Ver Notebook ➡️](https://github.com/Mitchell-Mirano/sorix/blob/develop/examples/nn/1-regression.ipynb) |
| **Capas NN** | Uso de módulos de capas de redes neuronales. | [Ver Notebook ➡️](https://github.com/Mitchell-Mirano/sorix/blob/develop/examples/basics/2-layers.ipynb) |

👉 **Más ejemplos:** Encuentra todos los casos de uso y tutoriales en la carpeta [`/examples`](https://github.com/Mitchell-Mirano/sorix/tree/main/examples) del repositorio.

-----

## 🚧 Estado del Proyecto

Sorix se encuentra en **desarrollo activo**. Estamos trabajando constantemente en la ampliación de funcionalidades clave:

  * Integración de más capas de redes neuronales esenciales.
  * Optimización y mejora del soporte para **GPU** a través de CuPy.
  * Extensión de la funcionalidad del motor `autograd`.

### ¡Contribuye\!

Agradecemos cualquier contribución de la comunidad. Puedes ayudar al proyecto de las siguientes maneras:

  * Reportando errores (Issues).
  * Añadiendo nuevas funcionalidades (Pull Requests).
  * Mejorando esta documentación.
  * Escribiendo pruebas unitarias.

-----

## 🔗 Enlaces Importantes

| Recurso | Enlace |
| :--- | :--- |
| **PyPI Package** | [Ver en PyPI](https://pypi.org/project/sorix/) |
| **Código Fuente** | [GitHub Repository](https://github.com/Mitchell-Mirano/sorix) |

-----