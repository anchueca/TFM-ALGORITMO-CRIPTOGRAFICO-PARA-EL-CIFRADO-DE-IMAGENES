# TFM - Algoritmo Criptográfico para el Cifrado de Imágenes

Este repositorio contiene el código desarrollado en el Trabajo de Fin de Máster titulado:

**Diseño e Implementación de un Algoritmo Criptográfico para el Cifrado de Imágenes Basado en Modelos Caóticos y Autómatas Celulares**

## Descripción

El algoritmo implementa un sistema de cifrado de imágenes utilizando:

- Dinámicas caóticas (modelo coseno-coseno).
- Autómatas celulares elementales (regla 30).
- Paralelización de alto rendimiento mediante CUDA (PyCUDA).

El objetivo es maximizar la seguridad y el rendimiento mediante técnicas criptográficas no lineales y estructuras altamente paralelizables, todo sobre arquitectura GPU.

## Características

- Cifrado por permutación y flujo
- Generación de claves pseudoaleatorias a partir de contraseñas mediante SHA-512
- Permutación de filas, columnas y bloques basada en autómatas y caos
- XOR con flujo generado caóticamente para difundir la intensidad de píxeles
- Implementación optimizada en Python + PyCUDA

## Requisitos

- Python 3.8+
- PyCUDA
- NumPy
- OpenCV
- Matplotlib

Instalación recomendada:
```bash
pip install -r requirements.txt
```