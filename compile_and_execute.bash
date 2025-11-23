#!/bin/bash

make -j 8 && ./cuda/bin/cipher.out ./repositorio/set3/lena3.jpg ./cuda/bin/salida.tif password $1 1 1 8 2 50 50 3.9 && ./cuda/bin/cipher.out ./cuda/bin/salida.tif ./cuda/bin/salidaC.tif password $1 0 0 8 2 50 50 3.9