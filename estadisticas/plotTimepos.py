import matplotlib.pyplot as plt

# Datos de ejemplo (rellena con tus datos)
# Número de píxeles por imagen (ancho x alto)
pixels = [
    256*256,     # camera.tif (ejemplo)
    512*512*3,     # lena3.tif (ajusta a la resolución real)
    512*512,     # mandrill.tif
    768*512*3,     # monarch.tif (ejemplo)
    512*512*3,      # peppers3.tif
    1024*1024,     # camera.tif (ejemplo)
    2048*2048*3,     # lena3.tif (ajusta a la resolución real)
    2048*2048,     # mandrill.tif
    768*512*3,     # monarch.tif (ejemplo)
    2048*3072*3      # peppers3.tif
]

# Tiempos en segundos para cifrado paralelo (CUDA)
tiempos_paralelo = [
    0.0240,
    0.0909,     # lena3
    0.0910,     # mandrill
    0.1474,     # monarch
    0.0906,     # peppers3
    0.4574,     # camera2
    2.2800,     # lena33
    2.3200,     # peppers33
    5.0981,      # monarch2
    3.2612
]

# Tiempos en segundos para cifrado lineal (CPU)
tiempos_lineal = [
    5.7191,
    21.2180,    # lena3
    21.4926,    # mandrill
    31.5436,    # monarch
    24.0773,    # peppers3
    73.0487,    # camera2
    310.0840,   # lena33
    306.6879,   # peppers33
    489.7835,    # monarch2
    310.5458    # mandrill2
]

plt.figure(figsize=(10,6))
plt.scatter(pixels, tiempos_paralelo, marker='o', color='blue', label='Cifrado Paralelo (CUDA)')
plt.scatter(pixels, tiempos_lineal, marker='s', color='red', label='Cifrado Lineal (CPU)')

plt.xlabel('Número de píxeles')
plt.ylabel('Tiempo de cifrado (segundos)')
plt.title('Comparación de tiempos de cifrado entre CPU y CUDA')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
plt.show()