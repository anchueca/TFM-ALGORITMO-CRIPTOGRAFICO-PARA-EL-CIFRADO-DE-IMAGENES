import cv2
import numpy as np
import argparse
import os
import sys
import time
import matplotlib.pyplot as plt
import math
import hashlib
import struct
from CelularAutoma2d import CelularAutomata2d
from ElementalCelularAutomata import ElementalCelularAutomata
from modeloCaos import selectFunction
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda_code = SourceModule("""
//CUDA
__device__ float uno(float x, float r) {
    float t = r + 3.0f * x * x;
    return fabsf(cosf(3.14159265f * r * cosf(3.14159265f * t) * t));
}

__global__ void flow_encrypt_recursive(
    unsigned char *image,
    float *seeds,
    float *finals,
    int width,
    int height,
    float r
) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < height && x < width) {
        int idx = y * width + x;
        float xn = seeds[y];

        for (int i = 0; i <= x; ++i) {
            xn = uno(xn, r);
        }

        image[idx] ^= (unsigned char)(xn * 256.0f);

        // Almacenar el valor final de xn en la última columna
        if (x == width - 1) {
            finals[y] = xn;
        }
    }
}
                         
__global__ void permute_blocks_kernel(
    unsigned char *input,
    unsigned char *output,
    int *permutations,
    int block_height,
    int block_width,
    int image_rows,
    int image_cols,
    int num_blocks_row,
    int num_blocks_col,
    int channels
) {
    int block_size = block_height * block_width;
    int total_blocks = num_blocks_row * num_blocks_col;
    int total_threads = block_size * total_blocks;

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= total_threads) return;

    int block_id = tid / block_size;
    int pixel_id = tid % block_size;

    int block_row = block_id / num_blocks_col;
    int block_col = block_id % num_blocks_col;

    int perm_index = block_id * block_size + pixel_id;
    int src_pos = permutations[perm_index];

    int dst_y = block_row * block_height + pixel_id / block_width;
    int dst_x = block_col * block_width + pixel_id % block_width;

    int src_y = block_row * block_height + src_pos / block_width;
    int src_x = block_col * block_width + src_pos % block_width;

    for (int c = 0; c < channels; ++c) {
        if (dst_y < image_rows && dst_x < image_cols && src_y < image_rows && src_x < image_cols) {
            output[(dst_y * image_cols + dst_x) * channels + c] =
                input[(src_y * image_cols + src_x) * channels + c];
        }
    }
}
//CUDA
""")

flow_encrypt_recursive = cuda_code.get_function("flow_encrypt_recursive")
permute_blocks_kernel = cuda_code.get_function("permute_blocks_kernel")

def get_Hash(text,length_bytes):
    hash= hashlib.sha512(text.encode('utf-8'))
    hash_result = hash.digest()
    extended_hash = hash_result
    while len(extended_hash) < length_bytes:
        extended_hash += hashlib.sha512(extended_hash).digest()
    extended_hash = extended_hash[:length_bytes]
    return extended_hash

def plot_histograms(image): # para pruebas luego eliminar

    # Calcular el histograma
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # Normalizar el histograma
    hist /= hist.sum()

    # Graficar como barras
    plt.figure(figsize=(10, 6))
    plt.bar(range(256), hist[:, 0], color='gray', edgecolor='black')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia Normalizada')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def generate_password(initial_password, num_blocks, row_number, column_number, byte_precission_level, color_depth, flow_rounds):

    # Required lengths
    bytes_for_rows = row_number * color_depth // 8
    bytes_for_columns = column_number * color_depth // 8
    bytes_for_blocks = num_blocks * byte_precission_level
    bytes_for_flow = row_number * byte_precission_level

    # Total length
    length_bytes = bytes_for_rows + bytes_for_columns + bytes_for_blocks + bytes_for_flow * flow_rounds
    hash = get_Hash(initial_password, length_bytes)

    index = 0

    # rows
    n = bytes_for_rows
    rows_automata_hash = np.array([int(bit) for byte in hash[index:index+n] for bit in format(byte, '08b')])
    index += n

    # columns
    n = bytes_for_columns
    columns_automata_hash = np.array([int(bit) for byte in hash[index:index+n] for bit in format(byte, '08b')])
    index += n

    # Blocks
    n = bytes_for_blocks
    blocks_hash = [struct.unpack('>H', bytes(hash[index + i:index + i + 2]))[0] / 65535.0 for i in range(0, n, 2)]
    index += n

    # Flow
    flow_hash = []
    for r in range(flow_rounds):
        n = bytes_for_flow
        flow_hash.append([struct.unpack('>H', bytes(hash[index + i:index + i + 2]))[0] / 65535.0 for i in range(0, n, 2)])
        index += n
    
    return rows_automata_hash, columns_automata_hash, blocks_hash, flow_hash


def encrypt_image(image,model, password, rounds, show=0):

    gris_scale=False

    if len(image.shape) > 2:
        gris_scale=True
        image = unstack_image(image)
        show_image(image,"unstacked") if show>1 else None

    num_blocks= 256
    precission_level=2
    image_rows,image_columns = image.shape[:2]
    color_depth = 8
    row_password, column_password, block_password, flow_password = generate_password(password,num_blocks,image_rows,image_columns,precission_level,color_depth,rounds)

    # Calculate the number of rows and columns
    num_rows = int(math.ceil(math.sqrt(num_blocks)))  # Rows
    num_cols = int(math.ceil(num_blocks / num_rows))  # Columns

    # BLock size as square as possible
    block_height = image_rows // num_rows
    block_width = image_columns // num_cols

    block_data_lenght=np.prod(block_height*block_width)
    bloock_permutations= [generate_partiton(model,block_password[i],block_data_lenght ) for i in range(num_blocks)] # all blocks have the same size

    permutation_rows = generate_partition_from_automata(row_password,image_rows)
    permutation_columns = generate_partition_from_automata(column_password,image_columns)
    
    for round_number in range(rounds):

        for n in range(2):

            image = permute_image_rows(image,permutation_rows)
            show_image(image,"permuted rows") if show>2 else None

            image = permute_image_columns(image,permutation_columns)
            show_image(image,"permuted columns") if show>2 else None

            image = image = block_phase_parallel_cuda(image,num_rows,num_cols,block_height,block_width,bloock_permutations)
            show_image(image,"reconstructed blocks") if show>1 else None

        if round_number < rounds - 1:  # The last rounds do not flow encrypt
            actual_flow_password=flow_password[round_number]
            for n in range(4):
                image,actual_flow_password=flow_encrypt_image_cuda(image,actual_flow_password)
                show_image(image,"flow image") if show>1 else None
            plot_histograms(image) if show>1 else None

    if gris_scale:
        image = stack_image(image)
    show_image(image,"encrypted image") if show>0 else None

    return image




def flow_encrypt_image_cuda(image, seeds, r=0.6):
    h, w = image.shape[:2]
    image = image.astype(np.uint8).copy()
    seeds = np.array(seeds, dtype=np.float32)
    finals = np.zeros(h, dtype=np.float32)

    # Reservar memoria en GPU
    dev_image = cuda.mem_alloc(image.nbytes)
    dev_seeds = cuda.mem_alloc(seeds.nbytes)
    dev_finals = cuda.mem_alloc(finals.nbytes)

    # Copiar datos de entrada
    cuda.memcpy_htod(dev_image, image)
    cuda.memcpy_htod(dev_seeds, seeds)

    block = (16, 16, 1)
    grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1])

    # Ejecutar kernel
    encrypt_gpu = cuda_code.get_function("flow_encrypt_recursive")
    encrypt_gpu(
        dev_image, dev_seeds, dev_finals,
        np.int32(w), np.int32(h),
        np.float32(r),
        block=block, grid=grid
    )

    # Copiar resultados
    cuda.memcpy_dtoh(image, dev_image)
    cuda.memcpy_dtoh(finals, dev_finals)

    # Liberar memoria
    dev_image.free()
    dev_seeds.free()
    dev_finals.free()

    return image, finals

def unencrypt_image(image,model, password, rounds, show=0):

    gris_scale=False
    if len(image.shape) > 2:
        gris_scale=True
        image = unstack_image(image)
        show_image(image,"unstacked") if show>1 else None


    num_blocks= 256
    precission_level=2
    image_rows,image_columns = image.shape[:2]
    color_depth = 8
    row_password, column_password, block_password, flow_password = generate_password(password,num_blocks,image_rows,image_columns,precission_level,color_depth,rounds)
    flow_password.reverse()

    # Calculate the number of rows and columns
    num_rows = int(math.ceil(math.sqrt(num_blocks)))  # Rows
    num_cols = int(math.ceil(num_blocks / num_rows))  # Columns

    # BLock size as square as possible
    block_height = image_rows // num_rows
    block_width = image_columns // num_cols

    block_data_lenght=np.prod(block_height*block_width)
    bloock_permutations= [generate_partiton(model,block_password[i],block_data_lenght ) for i in range(num_blocks)] # all blocks have the same size
    bloock_permutations= [invert_permutation(permitation) for permitation in bloock_permutations] # all blocks have the same size

    permutation_columns = invert_permutation(generate_partition_from_automata(column_password,image_columns))
    permutation_rows = invert_permutation(generate_partition_from_automata(row_password,image_rows))
    
    for round_number in range(rounds):

        if round_number != 0:  # The last rounds do not flow encrypt
            actual_flow_password=flow_password[round_number]
            for n in range(4):
                image,actual_flow_password=flow_encrypt_image_cuda(image,actual_flow_password)
                show_image(image,"flow image") if show>1 else None

        for n in range(2):
            image = block_phase_parallel_cuda(image,num_rows,num_cols,block_height,block_width,bloock_permutations)
            show_image(image,"reconstructed blocks") if show>1 else None
            
            image = permute_image_columns(image,permutation_columns)
            
            image = permute_image_rows(image,permutation_rows)
            show_image(image,"unpermuted") if show>1 else None

    if gris_scale:
        image = stack_image(image)
    show_image(image,"unencrypted image") if show>0 else None

    return image

def block_phase_parallel_cuda(image, num_rows, num_cols, block_height, block_width, block_permutations):
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    total_blocks = num_rows * num_cols
    block_size = block_height * block_width

    image_in = image.astype(np.uint8).ravel()
    image_out = np.empty_like(image_in)

    # Permutations: must be 1D array of shape (num_blocks * block_size)
    flat_permutations = np.concatenate(block_permutations).astype(np.int32)

    # Allocate device memory
    d_input = cuda.mem_alloc(image_in.nbytes)
    d_output = cuda.mem_alloc(image_out.nbytes)
    d_permutations = cuda.mem_alloc(flat_permutations.nbytes)

    cuda.memcpy_htod(d_input, image_in)
    cuda.memcpy_htod(d_permutations, flat_permutations)

    threads_per_block = 256
    grid_dim = (block_size + threads_per_block - 1)

    permute_blocks_kernel(
        d_input,
        d_output,
        d_permutations,
        np.int32(block_height),
        np.int32(block_width),
        np.int32(h),
        np.int32(w),
        np.int32(num_rows),
        np.int32(num_cols),
        np.int32(channels),
        block = (threads_per_block, 1, 1),
        grid = (grid_dim, 1)
    )

    cuda.memcpy_dtoh(image_out, d_output)
    return image_out.reshape((h, w) if channels == 1 else (h, w, channels))
            

def permute_image_rows(image, permutation):
    permuted_image = np.empty_like(image)
    for i, idx in enumerate(permutation):
        permuted_image[i] = image[idx]
    return permuted_image

def permute_image_columns(image, permutation):
    permuted_image = np.empty_like(image)
    for j, idx in enumerate(permutation):
        permuted_image[:, j] = image[:, idx]
    return permuted_image

def permute_box(box, permutation):
    """
    Permute the rows and columns of a box (2D array) based on a given permutation.
    """
    permuted_box = np.empty_like(box)
    h,w = box.shape
    
    # Permute rows
    for i, idx in enumerate(permutation):
        h_normalized= idx // w
        w_normalized = idx % w
        permuted_box[h_normalized][w_normalized] = box[h_normalized][w_normalized]    
    
    return permuted_box

def invert_permutation(permutation):
    inverted_permutation = np.empty_like(permutation)
    
    for i, idx in enumerate(permutation):
        inverted_permutation[idx] = i
        
    return inverted_permutation

def generate_partition_from_automata(state,lenght):
    '''Generate a permutation from an automata'''

    automata = ElementalCelularAutomata(state,len(state),30)

    automata.step_cuda(100)

    number_list = automata.convert_to_int()[:lenght]

    permutation = getPermutation(number_list)

    return permutation

def getPermutation(source):
    ''' Generate the partititon from a list of values'''
    if type(source[0]) is int: # case not numbered
        source = [ [n,value] for n,value in enumerate(source) ]

    sorted_list = sorted(source, key=lambda x: x[1])
    permutation_list = [sorted_list[n][0] for n in range(len(sorted_list))]
    
    return permutation_list


def generate_partiton(model,seed,length):
    '''Generates a permutation list based on the chaotic model.
    Args:
        seed: The initial value for the chaotic model.
        lenght: The maximum value for the permutation.
    Returns:
        A list of integers representing the permutation.
    '''
    xn = seed
    xn_list = []
    
    for n in range(20):
        xn = model(xn)
    
    for n in range(length):
        xn = model(xn)
        xn_list.append((n, xn))

    permutation_list = getPermutation(xn_list)
    
    return permutation_list

            
def unstack_image(image):
    '''Unstacks an image into its RGB channels and concatenates them horizontally.
    Args:
        image: The input image to unstack.
    Returns:        A concatenated image with the red, green, and blue channels side by side.
    '''
    b,g,r = cv2.split(image)

    concatenated = np.hstack((r, g, b))

    return concatenated


def stack_image(concatenated):
    _, width = concatenated.shape[:2]
    width_per_channel = width // 3
    
    img_r = concatenated[:, :width_per_channel]
    img_g = concatenated[:, width_per_channel:2*width_per_channel]
    img_b = concatenated[:, 2*width_per_channel:]
    
    if len(concatenated.shape) == 3: # If the image is in color
        img_r = np.max(img_r, axis=2)
        img_g = np.max(img_g, axis=2)
        img_b = np.max(img_b, axis=2)
    
    # Build the image from channels
    reconstructed_image = cv2.merge((img_b, img_g, img_r))
    
    return reconstructed_image

def show_image(image, title='Imagen'):
    '''Shows a image using OpenCV
    Args:
        image: The image to show
        title: The title of the window
    '''
    cv2.imshow(title, image)
    cv2.waitKey(0)

def main():
    parser = argparse.ArgumentParser(description="Cifrado de imagen")
    
    # Input arguments
    parser.add_argument("model", help="modelo")
    parser.add_argument("input_image", help="Ruta de la imagen a cifrar")
    #parser.add_argument("output_image", help="Ruta donde guardar la imagen cifrada")
    parser.add_argument("password", type=str, help="Contraseña para el cifrado")
    parser.add_argument("rounds", type=int, help="Number of rounds")
    parser.add_argument("show", type=int, help="Show data")
    parser.add_argument("other_args", nargs=argparse.REMAINDER, help="Otros argumentos para el cifrado")

    args = parser.parse_args()

    # Berify if the image exists
    if not os.path.exists(args.input_image):
        print(f"Error: La imagen de entrada '{args.input_image}' no existe.")
        sys.exit(1)

    model= selectFunction(args.model)[0]

    image = cv2.imread(args.input_image)

    if np.all(image[:, :, 0] == image[:, :, 1]) and np.all(image[:, :, 1] == image[:, :, 2]):
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    start_time = time.time()
    encrypted_image = encrypt_image(image,model, args.password, args.rounds, args.show)
    end_time = time.time()
    print(f"Time: {end_time - start_time} s")

    start_time = time.time()
    unencrypted_image = unencrypt_image(encrypted_image,model, args.password, args.rounds, args.show)
    end_time = time.time()
    print(f"Time: {end_time - start_time} s")

if __name__ == "__main__":
    main()

