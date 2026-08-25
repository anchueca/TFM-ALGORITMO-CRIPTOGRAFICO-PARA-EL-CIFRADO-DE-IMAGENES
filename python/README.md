# Herramientas de Análisis y Criptografía (Python)

Este directorio contiene todas las herramientas analíticas, matemáticas y estadísticas necesarias para estudiar la dinámica caótica del sistema y validar el esquema criptográfico implementado en CUDA.

Cada script dispone de su **propia carpeta dedicada** con un ejecutable Bash (`run.sh`) para realizar ejecuciones rápidas e interactivas empleando parámetros por defecto. El script `stats.py` se mantiene directamente en la raíz de `python/`.

---

## 1. Mapas Unidimensionales (1D)
Scripts utilizados para la exploración teórica y análisis aislado de funciones caóticas clásicas (Logístico, Tienda, Seno, Hénon, Uno).

- **`chaos_generator/`**:
  - `Chaos_Generator.py`: Diccionario central que define los mapas caóticos matemáticos.
  - `run.sh`: Simula y compara gráficamente los mapas caóticos.
- **`bifurcacion/`**:
  - `bifurcacion.py`: Generador de diagramas de bifurcación simples para los mapas 1D.
  - `run.sh`: Genera de forma rápida el diagrama de bifurcación por defecto.
- **`lyapunov/`**:
  - `lyapunov.py`: Calculadora de exponentes de Lyapunov para mapas 1D.
  - `run.sh`: Ejecuta y muestra la curva del exponente de Lyapunov.
- **`plot/`**:
  - `plot.py`: Herramienta visual simple para demostrar la alta sensibilidad a las condiciones iniciales (Efecto Mariposa).
  - `run.sh`: Muestra la gráfica interactiva de sensibilidad a condiciones iniciales.

---

## 2. Coupled Map Lattice (CML)
Scripts fundamentales que modelan, simulan y analizan la dinámica acoplada real utilizada en el algoritmo de cifrado.

- **`coupled_map/`**:
  - `coupled_map.py`: **[NÚCLEO]** Lógica matemática del CML y Autómata Celular de 16-bits.
  - `run.sh`: Simula las celdas acopladas y muestra su evolución temporal.
- **`coupled_lyapunov/`**:
  - `coupled_lyapunov.py`: Cálculo del Exponente Máximo de Lyapunov (LCE) del CML acoplado.
  - `run.sh`: Muestra la gráfica interactiva del espectro de Lyapunov.
- **`cml_analysis/`**:
  - `cml_analysis.py`: Análisis definitivo. Genera un panel dual (Bifurcación CML vs Exponente de Lyapunov).
  - `run.sh`: Ejecuta la herramienta de análisis completo y muestra el panel interactivo.
- **`cml_spectrum/`**: **[NUEVO]**
  - `cml_spectrum.py`: Espectro completo de $N$ exponentes de Lyapunov $(\lambda_1 \dots \lambda_N)$ mediante descomposición QR de la matriz Jacobiana, **Entropía de Kolmogorov-Sinai ($h_{KS}$)** y **Dimensión de Kaplan-Yorke ($D_{KY}$)**.
  - `run.sh`: Muestra la gráfica del espectro completo y la suma acumulada de los exponentes.
- **`coupled_lyapunov_diagram/`**:
  - `coupled_lyapunov_diagram.py`: Generador de mapas de calor 2D ($r$ vs $\epsilon$).
  - `run.sh`: Muestra el diagrama 2D de Lyapunov.
- **`cml_evolution/`**:
  - `cml_evolution.py`: Graficador del espacio-tiempo (Spacetime plot).
  - `run.sh`: Muestra la evolución espacio-tiempo de las celdas CML.

---

## 3. Criptoanálisis Avanzado y Testing Estadístico
Herramientas para someter a prueba la seguridad computacional de las secuencias generadas y del cifrado de imágenes/video con rigor matemático.

- **`nist_tests/`**:
  - `nist_tests.py`: Suite **NIST SP 800-22** (15 tests estadísticos). Valida la aleatoriedad del CML y CA.
  - `run.sh`: Ejecuta la batería de pruebas NIST y muestra una gráfica de barras interactiva con los resultados p-value.
- **`local_entropy/`**: **[NUEVO]**
  - `local_entropy.py`: Prueba formal de **Entropía Local de Shannon (LSE)** basada en la norma de Wu et al. ($k=30$ bloques desvinculados de $1936$ píxeles, $\mu=7.902486$) con contrastes de hipótesis a $\alpha = 0.05, 0.01, 0.001$.
  - `run.sh`: Genera el mapa de calor espacial LSE y la evaluación estadística.
- **`key_sensitivity/`**: **[NUEVO]**
  - `key_sensitivity.py`: Evaluación cuantitativa y visual de la **sensibilidad a la clave en cifrado y descifrado** ante perturbaciones extremas ($\Delta k = 10^{-15}$).
  - `run.sh`: Genera la gráfica comparativa multidimensional.
- **`differential_analysis/`**: **[NUEVO]**
  - `differential_analysis.py`: Test diferencial exhaustivo de **NPCR y UACI** con límites críticos teóricos $N^*_{\alpha}$ y $(U^{*-}_{\alpha}, U^{*+}_{\alpha})$, evaluando sensibilidad ante el cambio de un único píxel y baterías de imágenes sintéticas extremas.
  - `run.sh`: Ejecuta la batería diferencial y representa el gráfico con umbrales teóricos.
- **`stats.py`**:
  - Herramienta de auditoría global de imágenes cifradas (Entropía de Shannon global y local, NPCR, UACI, correlación de píxeles, histogramas, benchmarks contra AES/ASCON). Permanece en la raíz de `python/`.

---

### Uso de los Ejecutables Bash
Cada subcarpeta contiene un script `run.sh` ejecutable. Puedes iniciarlo directamente:

```bash
# Ejemplo 1: Calcular espectro de Lyapunov completo y dimensión Kaplan-Yorke
cd python/cml_spectrum
./run.sh

# Ejemplo 2: Test de Entropía Local de Shannon en una imagen
cd python/local_entropy
./run.sh --image /ruta/a/imagen_cifrada.png

# Ejemplo 3: Test de sensibilidad a la clave (Δk = 10^-15)
cd python/key_sensitivity
./run.sh
```
