#include "../include/encryption.cuh"

__host__ void encrypt_image(cv::Mat image, const std::string& password, int rounds, int verbose, bool encrypt) {
    
    // Setting parameters
    const size_t block_size = 32;
    const size_t precision_level = 2; // bytes
    const size_t automata_steps = 50;
    // For now we assume the image dimensions are multiples of block_size
    const size_t num_blocks_per_row = image.rows / block_size + (image.rows % block_size != 0);
    const size_t num_blocks_per_col = image.cols / block_size + (image.cols % block_size != 0);
    const size_t num_blocks = num_blocks_per_row * num_blocks_per_col;
    const size_t block_data_length = block_size*block_size;

    const std::vector<std::vector<unsigned char>> password_segments = calculate_password(password, num_blocks, precision_level, rounds, image.rows, image.cols);

    std::cout<< "Block size: " << block_size << std::endl;
    std::cout<< "Num blocks per row: " << num_blocks_per_row << std::endl;
    std::cout<< "Num blocks per col: " << num_blocks_per_col << std::endl;
    std::cout<< "Num blocks: " << num_blocks << std::endl;
    std::cout<< "Block data length: " << block_data_length << std::endl;
    std::cout<< "Password segment size: " << password_segments[3].size() << std::endl;
    std::cout<< image.rows << "x" << image.cols << std::endl;

    //Automatas
    printf("Generating row and column permutations using Elemental Cellular Automata...");
    ElementalCelularAutomata automata(password_segments[1],image.cols* precision_level * 8, 30);
    const std::vector<ElementalCelularAutomata*> cols_automata = {&automata};
    unsigned int* d_permutation_cols = generate_automata_permutations(cols_automata,automata_steps,image.cols);

    ElementalCelularAutomata automata1(password_segments[0],image.rows* precision_level * 8, 30);
    const std::vector<ElementalCelularAutomata*> rows_automata = {&automata1};
    unsigned int* d_permutation_rows = generate_automata_permutations(rows_automata,automata_steps,image.rows);
    printf("Done.\n");
    //Generate permutations
    printf("Generating blocks permutations using Chaotic function...\n");
    unsigned int* d_permutations =
        generate_flow_permutations(password_segments[2],block_data_length, num_blocks);
    unsigned char* d_image = nullptr;
    unsigned char* d_image_out = nullptr;
    const size_t img_size = image.total() * image.elemSize();

    cudaMalloc(&d_image, img_size);
    cudaMalloc(&d_image_out, img_size);

    cudaMemcpy(d_image, image.data, img_size, cudaMemcpyHostToDevice);

    if(encrypt){
        encryption_process(&d_image, &d_image_out,d_permutation_rows,d_permutation_cols, d_permutations, image.cols, image.rows,password_segments[3],block_size,rounds);
    }
    else{
        inverse_permutations(&d_permutation_cols, image.cols,1);
        inverse_permutations(&d_permutation_rows, image.rows,1);
        inverse_permutations(&d_permutations,block_data_length, num_blocks);

        unencryption_process(&d_image, &d_image_out,d_permutation_rows,d_permutation_cols, d_permutations, image.cols, image.rows,password_segments[3],block_size,rounds);
    }

    cudaMemcpy(image.data, d_image, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_permutation_cols);
    cudaFree(d_permutation_rows);
    cudaFree(d_permutations);

    cudaFree(d_image);
    cudaFree(d_image_out);
}

void encryption_process(unsigned char** d_image, unsigned char** d_image_out, unsigned int* d_permutation_rows, unsigned int* d_permutation_cols, unsigned int* d_permutation_blocks, size_t cols, size_t rows, std::vector<unsigned char> flow_seeds, size_t block_size, size_t rounds){
    unsigned char* temp = nullptr;

    std::cout << "Starting encryption with " << rounds << " rounds." << std::endl;
    for (size_t i=0;i<rounds;i++){
        for(size_t j=0; j<2;j++){//Each round two permtations
            //Rows an columns
            rows_and_columns_permutation(*d_image,*d_image_out,d_permutation_rows,d_permutation_cols, cols, rows, false);
            
            //Block
            block_phase_permutation(*d_image,*d_image_out, d_permutation_blocks, cols, rows, block_size);
            
            temp = *d_image;
            *d_image = *d_image_out;
            *d_image_out = temp;
        }

        flow_encrypt(*d_image, *d_image_out, flow_seeds, cols, rows, 3.999,1);
        temp = *d_image;
        *d_image = *d_image_out;
        *d_image_out = temp;
        
    }
}

void unencryption_process(unsigned char** d_image, unsigned char** d_image_out, unsigned int* d_permutation_rows, unsigned int* d_permutation_cols, unsigned int* d_permutation_blocks, size_t cols, size_t rows, std::vector<unsigned char> flow_seeds, size_t block_size, size_t rounds){
    unsigned char* temp = nullptr;

    std::cout << "Starting unencryption with " << rounds << " rounds." << std::endl;
    for (size_t i=0;i<rounds;i++){
        flow_encrypt(*d_image, *d_image_out, flow_seeds, cols, rows, 3.999,1);
    
        temp = *d_image;
        *d_image = *d_image_out;
        *d_image_out = temp;
        
        for(size_t j=0; j<2;j++){//Each round two permtations
            //Block
            block_phase_permutation(*d_image,*d_image_out, d_permutation_blocks, cols, rows, block_size);
            
            temp = *d_image;
            *d_image = *d_image_out;
            *d_image_out = temp;

            //Rows an columns
            rows_and_columns_permutation(*d_image,*d_image_out,d_permutation_rows,d_permutation_cols, cols, rows, true);
            
        }
        
    }
}