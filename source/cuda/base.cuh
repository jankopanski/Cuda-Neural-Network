#ifndef BASE_CUH_
#define BASE_CUH_

#include <string>


const int layer_num = 6;
const int layer_sizes[layer_num] = {4096, 8192, 6144, 3072, 1024, 62};
const int feature_num = layer_sizes[0];
const float initial_weight = 1.0;

/* Setting const_labels_size to 0 stores labels in global memory.
 * Setting const_labels_size > 0 stores labels in constant memory,
 * constant memory is bounded up to 16384 labels. */
const int const_labels_size = 16384;
const int block_dim_1D = 256;
const int grid_dim_1D = 28;
const int block_size = 32;
const dim3 block_dim_2D = dim3(block_size, block_size);
const dim3 grid_dim_2D = dim3(8, 7);


struct Arguments {
    std::string training_data_path;
    float epsilon{};
    float learning_rate{};
    int epochs{};
    bool random{};
    int data_size{};
};

struct HostStructures {
	int *labels;
	float *input_layer;
	float *evaluate_helper;
	int correct_predictions;
};

struct DeviceStructures {
	int *labels;
	float *weights[layer_num - 1];
	float *biases[layer_num - 1];
	float *layers_outputs[layer_num];
	float *errors[layer_num - 1];
	float *weights_partial_derivatives[layer_num - 1];
	float *weights_gradients[layer_num - 1];
	float *biases_gradients[layer_num - 1];
	float *evaluate_helper;
	int *correct_predictions;
};


#endif /* BASE_CUH_ */
