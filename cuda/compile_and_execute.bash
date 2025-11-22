#!/bin/bash
make -j 4 && ./bin/cipher.out ../repositorio/set3/lena3.jpg ./bin/salida.tif password 3 1 1 8 2 20 10 && ./bin/cipher.out ./bin/salida.tif ./bin/salidaC.tif password 3 0 0 8 2 20 10