#include "../include/encryption.cuh"

__host__ void encrypt_image(cv::Mat image, const std::string& password, const EncryptionParams& params, bool verbose, bool encrypt) {
    
    // Setting parameters
    const size_t block_size = params.block_size;
    const size_t precision_level = params.precision_level;
    const size_t automata_steps = params.automata_steps;
    const size_t transition_length = params.transition_length;
    const size_t rounds = params.rounds;
    // For now we assume the image dimensions are multiples of block_size
    const size_t num_blocks_per_row = image.rows / block_size + (image.rows % block_size != 0);
    const size_t num_blocks_per_col = image.cols / block_size + (image.cols % block_size != 0);
    const size_t num_blocks = num_blocks_per_row * num_blocks_per_col;
    const size_t block_data_length = block_size*block_size;
    const size_t num_blocks_per_row = image.rows / block_size + (image.rows % block_size != 0);
    const size_t num_blocks_per_col = image.cols / block_size + (image.cols % block_size != 0);
    const size_t num_blocks = num_blocks_per_row * num_blocks_per_col;
    const size_t block_data_length = block_size*block_size;

    auto start = std::chrono::high_resolution_clock::now();
    const std::vector<std::vector<unsigned char>> password_segments = calculate_password(password, num_blocks, precision_level, rounds, image.rows, image.cols);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end - start;
    if(verbose)std::cout<<"Password generation time: "<< time.count()<< " s"<<std::endl;

    if(verbose){
        std::cout<< "=== Encryption parameters ===" << std::endl;
        std::cout<< "\tPrecision level: " << precision_level << std::endl;
        std::cout<< "\tAutomata steps: " << automata_steps << std::endl;
        std::cout<< "\tTransition length: " << transition_length << std::endl;
        std::cout<< "\tBlock size: " << block_size << std::endl;
        std::cout<< "\tNum blocks per row: " << num_blocks_per_row << std::endl;
        std::cout<< "\tNum blocks per col: " << num_blocks_per_col << std::endl;
        std::cout<< "\tNum blocks: " << num_blocks << std::endl;
        std::cout<< "\tBlock data length: " << block_data_length << std::endl;
    }

    //Automatas
    if(verbose)std::cout<<"Generating row and column permutations using Elemental Cellular Automata..."<<std::endl;
    ElementalCelularAutomata automata(password_segments[1],image.cols* precision_level * 8, 30);
    const std::vector<ElementalCelularAutomata*> cols_automata = {&automata};

    start = std::chrono::high_resolution_clock::now();
    unsigned int* d_permutation_cols = generate_automata_permutations(cols_automata,automata_steps,image.cols);
    end = std::chrono::high_resolution_clock::now();
    time = end - start;
    if(verbose)std::cout<<"\t\tgenerate_automata_permutations (cols) time: "<< time.count()<< " s"<<std::endl;

    ElementalCelularAutomata automata1(password_segments[0],image.rows* precision_level * 8, 30);
    const std::vector<ElementalCelularAutomata*> rows_automata = {&automata1};

    start = std::chrono::high_resolution_clock::now();
    unsigned int* d_permutation_rows = generate_automata_permutations(rows_automata,automata_steps,image.rows);
    end = std::chrono::high_resolution_clock::now();
    time = end - start;
    if(verbose)std::cout<<"\t\tgenerate_automata_permutations (rows) time: "<< time.count()<< " s"<<std::endl;

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
    if(verbose)std::cout<<("Generating blocks permutations using Chaotic function...")<<std::endl;

    start = std::chrono::high_resolution_clock::now();
    unsigned int* d_permutations =
    generate_flow_permutations(password_segments[2],block_data_length, num_blocks,transition_length);
    end = std::chrono::high_resolution_clock::now();
    time = end - start;
    if(verbose)std::cout<<"\t\tgenerate_flow_permutations (blocks) time: "<< time.count()<< " s"<<std::endl;

    unsigned char* d_image = nullptr;
    unsigned char* d_image_out = nullptr;
    const size_t img_size = image.total() * image.elemSize();

    cudaMalloc(&d_image, img_size);
    cudaMalloc(&d_image_out, img_size);

    cudaMemcpy(d_image, image.data, img_size, cudaMemcpyHostToDevice);

    if(encrypt){
        encryption_process(&d_image, &d_image_out,d_permutation_rows,d_permutation_cols, d_permutations, image.cols, image.rows,password_segments[3],block_size,rounds,verbose);
    }
    else{
        start = std::chrono::high_resolution_clock::now();
        inverse_permutations(&d_permutation_cols, image.cols,1);
        end = std::chrono::high_resolution_clock::now();
        time = end - start;
        if(verbose)std::cout<<"\tinverse rows time: "<< time.count()<< " s"<<std::endl;

        start = std::chrono::high_resolution_clock::now();
        inverse_permutations(&d_permutation_rows, image.rows,1);
        end = std::chrono::high_resolution_clock::now();
        time = end - start;
        if(verbose)std::cout<<"\tinverse rows time: "<< time.count()<< " s"<<std::endl;

        start = std::chrono::high_resolution_clock::now();
        inverse_permutations(&d_permutations,block_data_length, num_blocks);
        end = std::chrono::high_resolution_clock::now();
        time = end - start;
        if(verbose)std::cout<<"\tinverse blocks time: "<< time.count()<< " s"<<std::endl;
        
        unencryption_process(&d_image, &d_image_out,d_permutation_rows,d_permutation_cols, d_permutations, image.cols, image.rows,password_segments[3],block_size,rounds);
    }


    cudaMemcpy(image.data, d_image, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_permutation_cols);
    cudaFree(d_permutation_rows);
    cudaFree(d_permutations);

    cudaFree(d_permutation_cols);
    cudaFree(d_permutation_rows);
    cudaFree(d_permutations);

    cudaFree(d_image);
    cudaFree(d_image_out);
}

void encryption_process(unsigned char** d_image, unsigned char** d_image_out, unsigned int* d_permutation_rows, unsigned int* d_permutation_cols, unsigned int* d_permutation_blocks, size_t cols, size_t rows, std::vector<unsigned char> flow_seeds, size_t block_size, size_t rounds, bool verbose){
    unsigned char* temp = nullptr;

    if(verbose)std::cout << "Starting encryption with " << rounds << " rounds." << std::endl;
    for (size_t i=0;i<rounds;i++){
        auto start = std::chrono::high_resolution_clock::now();
        
        for(size_t j=0; j<2;j++){//Each round two permtations
            //Rows an columns
            auto start1 = std::chrono::high_resolution_clock::now();
            rows_and_columns_permutation(*d_image,*d_image_out,d_permutation_rows,d_permutation_cols, cols, rows, false);
            auto end1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time1 = end1 - start1;
            std::cout<<"\t\t\t\trows_and_columns_permutation("<< i << ")"<< " time: "<< time1.count()<< " s"<<std::endl;
            //Block
            start = std::chrono::high_resolution_clock::now();
            block_phase_permutation(*d_image,*d_image_out, d_permutation_blocks, cols, rows, block_size);
            end1 = std::chrono::high_resolution_clock::now();
            time1 = end1 - start1;
            std::cout<<"\t\t\t\tblock_phase_permutation("<< i << ")"<< " time: "<< time1.count()<< " s"<<std::endl;
            temp = *d_image;
            *d_image = *d_image_out;
            *d_image_out = temp;
        }
        auto start1 = std::chrono::high_resolution_clock::now();
        flow_encrypt(*d_image, *d_image_out, flow_seeds, cols, rows, 3.999,1);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time1 = end1 - start1;
        std::cout<<"\t\t\t\tflow_encrypt("<< i << ")"<< " time: "<< time1.count()<< " s"<<std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time = end - start;
        std::cout<<"\t\t\tround("<< i << ")"<< " time: "<< time.count()<< " s"<<std::endl;
        
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