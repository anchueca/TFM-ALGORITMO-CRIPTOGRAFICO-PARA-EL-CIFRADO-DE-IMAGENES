# Herramientas de Análisis y Criptografía (Python)

Este directorio contiene todas las herramientas analíticas, matemáticas y estadísticas necesarias para estudiar la dinámica caótica del sistema y validar el esquema criptográfico implementado en CUDA.

Para mantener la integridad de las importaciones y la usabilidad, todos los scripts activos se encuentran en la raíz de este directorio, organizados lógicamente en las siguientes **tres categorías**.

---

## 1. Mapas Unidimensionales (1D)
Scripts utilizados para la exploración teórica y análisis aislado de funciones caóticas clásicas (Logístico, Tienda, Seno, Hénon).

- **`Chaos_Generator.py`**: Diccionario central que define los mapas caóticos matemáticos.
- **`bifurcacion.py`**: Generador de diagramas de bifurcación simples para los mapas 1D.
- **`lyapunov.py`**: Calculadora de exponentes de Lyapunov para mapas 1D mediante el algoritmo de su matriz Jacobiana.
- **`plot.py`**: Herramienta visual simple para demostrar la alta sensibilidad a las condiciones iniciales (Efecto Mariposa) simulando dos trayectorias contiguas.

---

## 2. Coupled Map Lattice (CML)
Scripts fundamentales que modelan, simulan y analizan la dinámica acoplada real utilizada en el algoritmo de cifrado.

- **`coupled_map.py`**: **[NÚCLEO]** Contiene la lógica matemática exacta del CML y del Autómata Celular Acoplado de 16-bits. Funciona como la versión Python 1:1 de los kernels de CUDA, incluyendo el acoplamiento bidireccional.
- **`coupled_lyapunov.py`**: Motor matemático para calcular la matriz Jacobiana a alta dimensión y extraer el Exponente Máximo de Lyapunov (LCE) del sistema CML acoplado.
- **`cml_analysis.py`**: Herramienta de análisis definitiva. Genera un panel dual que correlaciona directamente el Diagrama de Bifurcación del CML con la gráfica de su Exponente Máximo de Lyapunov en función del parámetro $r$.
- **`coupled_lyapunov_diagram.py`**: Generador de mapas de calor 2D. Evalúa la estabilidad del sistema explorando simultáneamente el parámetro de caos ($r$) y el parámetro de acoplamiento ($\epsilon$).
- **`cml_evolution.py`**: Graficador del espacio-tiempo (Spacetime plot). Permite visualizar cómo evolucionan e interactúan espacialmente todas las celdas del CML a lo largo de las iteraciones.

---

## 3. Criptoanálisis y Testing Estadístico
Herramientas para someter a prueba la seguridad computacional de las secuencias generadas y del cifrado de imágenes.

- **`nist_tests.py`**: Implementación completa de la suite **NIST SP 800-22** (15 tests estadísticos). Valida la aleatoriedad de secuencias generadas por:
  1. El sistema caótico continuo (CML).
  2. El Autómata Celular (CA Acoplado).
  3. El flujo real de bytes (Keystream) extraído directamente del ejecutable CUDA.
- **`stats.py`**: Herramienta de auditoría de imágenes cifradas. Calcula métricas clave (Entropía de Shannon, NPCR, UACI, correlación de píxeles adyacentes, visualización de histogramas) y compara automáticamente los resultados y el rendimiento contra sistemas estándar como AES-256-CTR o ASCON.

---

### Requisitos y Entorno
Es imprescindible ejecutar estos scripts dentro del entorno virtual configurado en la raíz del repositorio (`source .env/bin/activate`), ya que dependen fuertemente de bibliotecas científicas (`numpy`, `scipy`, `matplotlib`, etc.).
