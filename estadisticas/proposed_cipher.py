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

    # Cálculo de tamaños en bytes
    bytes_for_rows = row_number * color_depth // 8
    bytes_for_columns = column_number * color_depth // 8
    bytes_for_blocks = num_blocks * byte_precission_level
    bytes_for_flow = row_number * byte_precission_level

    # Longitud total de los bytes necesarios
    length_bytes = bytes_for_rows + bytes_for_columns + bytes_for_blocks + bytes_for_flow * flow_rounds
    hash = get_Hash(initial_password, length_bytes)  # Asumiendo que get_Hash está definida

    index = 0

    # Procesar filas
    n = bytes_for_rows
    rows_automata_hash = np.array([int(bit) for byte in hash[index:index+n] for bit in format(byte, '08b')])
    index += n  # Actualizar el índice

    # Procesar columnas
    n = bytes_for_columns
    columns_automata_hash = np.array([int(bit) for byte in hash[index:index+n] for bit in format(byte, '08b')])
    index += n  # Actualizar el índice

    # Procesar bloques (float normalizado en el rango [0, 1])
    n = bytes_for_blocks
    blocks_hash = [struct.unpack('>H', bytes(hash[index + i:index + i + 2]))[0] / 65535.0 for i in range(0, n, 2)]
    index += n  # Actualizar el índice

    # Procesar flujo (también como un float normalizado)
    flow_hash = []
    for r in range(flow_rounds):
        n = bytes_for_flow
        flow_hash.append([struct.unpack('>H', bytes(hash[index + i:index + i + 2]))[0] / 65535.0 for i in range(0, n, 2)])
        index += n  # Actualizar el índice
    
    return rows_automata_hash, columns_automata_hash, blocks_hash, flow_hash




def encrypt_image(image,model, password, rounds, show=0):

    gris_scale=False

    if len(image.shape) > 2:
        gris_scale=True
        #image = unstack_image(image)
        show_image(image,"unstacked") if show>1 else None

    num_blocks= 64
    precission_level=2
    rows,columns = image.shape[:2]
    color_depth = 8
    row_password, column_password, block_password, flow_password = generate_password(password,num_blocks,rows,columns,precission_level,color_depth,rounds)

    permutation_rows = generate_partition_from_automata(row_password,image.shape[0])
    permutation_columns = generate_partition_from_automata(column_password,image.shape[1])
    
    for round_number in range(rounds):
        image = permute_image_rows(image,permutation_rows)
        show_image(image,"permuted rows") if show>1 else None

        
        image = permute_image_columns(image,permutation_columns)
        show_image(image,"permuted columns") if show>1 else None

        blocks= get_image_blocks(image,num_blocks)
        show_image(blocks[0][0])
        show_image(blocks[0][1])
        show_image(blocks[-1][-1])

        block_data_lenght=np.prod(blocks[0][0].shape[:2])
        block_permutations= [generate_partiton(model,block_password[i],block_data_lenght ) for i in range(num_blocks)] # all blocks have the same size

        blocks=permute_block_matrix(blocks,block_permutations)
        show_image(blocks[0][0])
        show_image(blocks[0][1])
        show_image(blocks[-1][-1])
        

        image = compose_image_from_blocks(blocks,image.shape,num_blocks)
        show_image(image,"reconstructed blocks") if show>1 else None

        if round_number < rounds - 1:  # The last rounds do not flow encrypt
            actual_flow_password=flow_password[round_number]
            for n in range(4):
                image,actual_flow_password=flow_encrypt_image(image,model,actual_flow_password)
                show_image(image,"flow image") if show>1 else None
            plot_histograms(image) if show>1 else None

    if gris_scale:
        image = stack_image(image)
    show_image(image,"encrypted image") if show>0 else None

    return image

def flow_encrypt_image(image,model, password):
    h,w = image.shape[:2]

    for i in range(h):
        image,password[i]=flow_encrypt_image_unit(image,model,i,w,password[i])
        
    return image,password

def flow_encrypt_image_unit(image,model,i,w, password):
    xn = password
    for j in range(w):
        xn = model(xn)
        image[i][j]= image[i][j] ^ int(xn*256)
    return image,xn

def unencrypt_image(image,model, password, rounds, show=0):

    gris_scale=False
    if len(image.shape) > 2:
        gris_scale=True
        image = unstack_image(image)
        show_image(image,"unstacked") if show>1 else None


    num_blocks= 256
    precission_level=2
    rows,columns = image.shape[:2]
    color_depth = 8
    row_password, column_password, block_password, flow_password = generate_password(password,num_blocks,rows,columns,precission_level,color_depth,rounds)
    flow_password.reverse()
    
    permutation_columns = invert_permutation(generate_partition_from_automata(column_password,image.shape[1]))
    permutation_rows = invert_permutation(generate_partition_from_automata(row_password,image.shape[0]))

    for round_number in range(rounds):

        if round_number != 0:  # The last rounds do not flow encrypt
            actual_flow_password=flow_password[round_number]
            for n in range(4):
                image,actual_flow_password=flow_encrypt_image(image,model,actual_flow_password)
                show_image(image,"flow image") if show>1 else None

        blocks= get_image_blocks(image,num_blocks)
        
        block_data_lenght=np.prod(blocks[0][0].shape[:2])
        bloock_permutations= [generate_partiton(model,block_password[i],block_data_lenght ) for i in range(num_blocks)] # all blocks have the same size
        bloock_permutations= [invert_permutation(permitation) for permitation in bloock_permutations] # all blocks have the same size
        blocks=permute_block_matrix(blocks,bloock_permutations)
        image = compose_image_from_blocks(blocks,image.shape,num_blocks)
        show_image(image,"reconstructed blocks") if show>1 else None
        
        image = permute_image_columns(image,permutation_columns)
        image = permute_image_rows(image,permutation_rows)
        show_image(image,"unpermuted") if show>1 else None

    if gris_scale:
        image = stack_image(image)
    show_image(image,"unencrypted image") if show>0 else None

    return image


def get_image_blocks(image,num_blocks=64):
    x,y = image.shape[:2]

    # Calculate the number of rows and columns
    num_rows = int(math.ceil(math.sqrt(num_blocks)))  # Rows
    num_cols = int(math.ceil(num_blocks / num_rows))  # Columns

    # BLock size as square as possible
    block_height = x // num_rows
    block_width = y // num_cols

    blocks = []
    
    # Split image into blocks
    for i in range(num_rows):
        row_blocks = []
        for j in range(num_cols):
            # begin and end of the block
            start_x = i * block_height
            end_x = (i + 1) * block_height if (i + 1) * block_height <= x else x
            
            start_y = j * block_width
            end_y = (j + 1) * block_width if (j + 1) * block_width <= y else y
            
            # Get the block
            block = image[start_x:end_x, start_y:end_y]
            row_blocks.append(block)
        blocks.append(row_blocks)

    blocks = np.array(blocks)
    return blocks

def permute_block_matrix(block_list,permutation_list):
    """
    Permute the blocks in a list based on a given permutation.
    """
    permuted_blocks = [ [permute_block(block, permutation) for block in block_row] for block_row, permutation in zip(block_list, permutation_list)]
    permuted_blocks = np.array(permuted_blocks)

    return permuted_blocks

def permute_block(block, permutation):
    """
    Permute the rows and columns of a block (2D array) based on a given permutation.
    """
    permuted_block = np.empty_like(block)
    h,w = block.shape[:2]
    
    # Permute rows
    for i, idx in enumerate(permutation):
        h_normalized= i // w
        w_normalized = i % w
        h_permutation_normalized= idx // w
        w_permutation_normalized = idx % w
        permuted_block[h_normalized][w_normalized] = block[h_permutation_normalized][w_permutation_normalized]    
    
    return permuted_block
            

def compose_image_from_blocks(blocks, image_shape, num_blocks=64):
    x, y = image_shape[:2]
    
    num_rows,num_cols= blocks.shape[:2]

    block_height = x // num_rows
    block_width = y // num_cols

    reconstructed_image = np.empty((x,y),dtype=np.uint8) if len(image_shape) == 2 else np.zeros((x,y,3), dtype=np.uint8)

    # Build the image
    for i in range(num_rows):
        for j in range(num_cols):
            
            start_x = i * block_height
            end_x = (i + 1) * block_height if (i + 1) * block_height <= x else x

            start_y = j * block_width
            end_y = (j + 1) * block_width if (j + 1) * block_width <= y else y

            # Put the block into the image
            reconstructed_image[start_x:end_x, start_y:end_y] = blocks[i][j]

    return reconstructed_image

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

    bit_depth= 8

    automata = ElementalCelularAutomata(state,len(state),30)

    automata.step(10)

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


def generate_partiton(model,seed,lenght):
    '''Generates a permutation list based on the logistic map.
    Args:
        seed: The initial value for the logistic map.
        lenght: The maximum value for the permutation.
    Returns:
        A list of integers representing the permutation.
    '''
    xn = seed
    b1 = 4.0
    xn_list = []
    
    for n in range(50):
        xn = model(xn,b1)
    
    for n in range(lenght):
        xn = model(xn,b1)
        xn_list.append([n, xn])

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
    height, width = concatenated.shape[:2]
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


