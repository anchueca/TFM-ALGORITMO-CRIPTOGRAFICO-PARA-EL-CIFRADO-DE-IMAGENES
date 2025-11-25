#!/bin/bash

make -j 8 && ./cuda/bin/cipher.out ./repositorio/set3/peppers3.tif ./cuda/bin/salida.tif password $1 1 8 4 50 50 3.9 1 && ./cuda/bin/cipher.out ./cuda/bin/salida.tif ./cuda/bin/salidaC.tif password $1 0 8 4 50 50 3.9 0