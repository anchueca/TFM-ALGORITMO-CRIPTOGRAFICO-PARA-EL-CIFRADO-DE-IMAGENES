#include "../include/encryption.cuh"

__host__ void encrypt_image(cv::Mat image, const std::string& password, int rounds, int verbose) {
    
    // Setting parameters
    const size_t block_size = 32;
    const size_t precision_level = 2; // 2 bytes
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
    ElementalCelularAutomata automata(password_segments[1],image.cols* precision_level * 8, 30);
    const std::vector<ElementalCelularAutomata*> container1 = {&automata};
    unsigned int* d_permutation_cols = generate_automata_permutations(container1,automata_steps,image.cols);

    ElementalCelularAutomata automata1(password_segments[0],image.rows* precision_level * 8, 30);
    const std::vector<ElementalCelularAutomata*> container = {&automata1};
    unsigned int* d_permutation_rows = generate_automata_permutations(container,automata_steps,image.rows);

    //Generate permutations
    unsigned int* permutations =
        generate_flow_permutations(password_segments[2],block_data_length, num_blocks);


    unsigned char* d_image = nullptr;
    unsigned char* d_image_out = nullptr;
    const size_t img_size = image.total() * image.elemSize();

    cudaMalloc(&d_image, img_size);
    cudaMalloc(&d_image_out, img_size);

    cudaMemcpy(d_image, image.data, img_size, cudaMemcpyHostToDevice);

    unsigned char* temp = nullptr;

    std::cout << "Starting encryption with " << rounds << " rounds." << std::endl;
    for (size_t i=0;i<rounds;i++){
        for(size_t j=0; j<2;j++){//Each round two permtations
            //Rows an columns
            rows_and_columns_permutation(d_image,d_image_out,d_permutation_rows,d_permutation_cols, image.cols, image.rows, false);
            
            //Block
            block_phase_permutation(d_image,d_image_out, permutations, image.cols, image.rows, block_size);
            
            temp = d_image;
            d_image = d_image_out;
            d_image_out = temp;
        }

        /*flow_encrypt(d_image, d_image_out, password_segments[3], image.cols, image.rows, 3.999,1);
        temp = d_image;
        d_image = d_image_out;
        d_image_out = temp;*/
        

    }
    cudaMemcpy(image.data, d_image, img_size, cudaMemcpyDeviceToHost);

    cudaFree(permutations);
    cudaFree(d_image);
    cudaFree(d_image_out);
}
