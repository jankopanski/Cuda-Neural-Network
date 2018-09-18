/*
 ============================================================================
 Name        : mlp.cu
 Author      : Jan Kopański
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <getopt.h>
#include <cuda.h>
#include <curand.h>
#include "base.cuh"
#include "kernels.cuh"
#include "debug.cuh"

using namespace std;


void parseArguments(int argc, char *argv[], Arguments &arguments) {
	if (argc != 11) {
		throw invalid_argument("Invalid number of command line arguments");
	}
    struct option options[] = {
            {"training_data", required_argument, nullptr, 1},
            {"epsilon",       required_argument, nullptr, 2},
            {"learning_rate", required_argument, nullptr, 3},
            {"epochs",        required_argument, nullptr, 4},
            {"random",		  required_argument, nullptr, 5},
            {nullptr,         0,                 nullptr, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1) {
        switch (opt) {
            case 1:
                arguments.training_data_path = optarg;
                break;
            case 2:
                arguments.epsilon = stof(optarg);
                break;
            case 3:
                arguments.learning_rate = stof(optarg);
                break;
            case 4:
                arguments.epochs = stoi(optarg);
                break;
            case 5:
            	if (!strcmp(optarg, "true")) {
            		arguments.random = true;
            		break;
            	}
            	else if (!strcmp(optarg, "false")) {
            		arguments.random = false;
            		break;
            	}
            default:
                throw invalid_argument("Invalid command line argument");
        }
    }
}

void feedforward(int data_size, DeviceStructures &device) {
	for (int i = 1; i < layer_num; i++) {
		compute_hidden_layer_outputs<<<grid_dim_2D, block_dim_2D>>> (
				layer_sizes[i - 1], layer_sizes[i], data_size,
				device.layers_outputs[i - 1], device.layers_outputs[i],
				device.weights[i - 1], device.biases[i - 1]);
	}
	compute_softmax_layer_activations<<<grid_dim_2D, block_dim_2D>>> (
			layer_sizes[layer_num - 2], layer_sizes[layer_num - 1], data_size,
			device.layers_outputs[layer_num - 2], device.layers_outputs[layer_num - 1],
			device.weights[layer_num - 2], device.biases[layer_num - 2]);
	compute_softmax_layer_outputs<<<grid_dim_1D, block_dim_1D>>> (
			layer_sizes[layer_num - 1], data_size, device.layers_outputs[layer_num - 1]);
}

void backpropagate(int data_size, float learning_rate, DeviceStructures &device) {
	const int stream_num = layer_num - 1;
	cudaStream_t stream[stream_num];
	for (int i = 0; i < stream_num; i++) {
		CUDA_CHECK_RETURN( cudaStreamCreate(&stream[i]) );
	}
	compute_output_layer_errors<<<grid_dim_1D, block_dim_1D>>> (
			layer_sizes[layer_num - 1], data_size, device.labels,
			device.layers_outputs[layer_num - 1], device.errors[layer_num - 2]);
	for (int i = layer_num - 3; i >= 0; i--) {
		compute_hidden_layer_errors<<<grid_dim_2D, block_dim_2D>>> (
				layer_sizes[i + 2], layer_sizes[i + 1], data_size,
				device.layers_outputs[i + 1], device.weights[i + 1],
				device.errors[i + 1], device.errors[i]);
	}
	for (int i = 1; i < layer_num; i++) {
		compute_weights_gradients<<<grid_dim_2D, block_dim_2D, 0, stream[i - 1]>>> (
				layer_sizes[i - 1], layer_sizes[i], data_size,
				device.layers_outputs[i - 1], device.errors[i - 1], device.weights_gradients[i - 1]);
	}
	for (int i = 1; i < layer_num; i++) {
		update_weights<<<grid_dim_2D, block_dim_2D, 0, stream[i - 1]>>> (
				learning_rate, layer_sizes[i - 1], layer_sizes[i],
				device.weights_gradients[i - 1], device.weights[i - 1]);
	}
	for (int i = 1; i < layer_num; i++) {
		compute_biases_gradients<<<grid_dim_1D, block_dim_1D, 0, stream[i - 1]>>> (
				layer_sizes[i], data_size, device.errors[i - 1], device.biases_gradients[i - 1]);
	}
	for (int i = 1; i < layer_num; i++) {
		int block_dim = min(layer_sizes[i], block_dim_1D);
		update_biases<<<1, block_dim, 0, stream[i - 1]>>> (
				learning_rate, layer_sizes[i], device.biases_gradients[i - 1], device.biases[i - 1]);
	}
	for (int i = 0; i < stream_num; i++) {
		CUDA_CHECK_RETURN( cudaStreamSynchronize(stream[i]) );
	}
	for (int i = 0; i < stream_num; i++) {
		CUDA_CHECK_RETURN( cudaStreamDestroy(stream[i]) );
	}
}

pair<float, float> evaluate(int data_size, HostStructures &host, DeviceStructures &device) {
	evaluate_loss<<<grid_dim_1D, block_dim_1D>>> (
			layer_sizes[layer_num - 1], data_size,
			device.labels, device.layers_outputs[layer_num - 1], device.evaluate_helper);
	CUDA_CHECK_RETURN( cudaMemcpy(host.evaluate_helper, device.evaluate_helper, grid_dim_1D * sizeof(float), cudaMemcpyDeviceToHost) );
	float loss = 0;
	for (int i = 0; i < grid_dim_1D; i++) {
		loss += host.evaluate_helper[i];
	}
	evaluate_accuracy<<<grid_dim_1D, block_dim_1D>>> (
			layer_sizes[layer_num - 1], data_size,
			device.labels, device.layers_outputs[layer_num - 1], device.correct_predictions);
	CUDA_CHECK_RETURN( cudaMemcpy(&host.correct_predictions, device.correct_predictions, sizeof(int), cudaMemcpyDeviceToHost) );
	float accuracy = static_cast<float>(host.correct_predictions) / static_cast<float>(data_size);
	return make_pair(accuracy, loss);
}

void train(HostStructures &host, DeviceStructures &device, Arguments &arguments) {
	float average_accuracy = 0;
	int epoch = 1;
	feedforward(arguments.data_size, device);
	for (; epoch <= arguments.epochs; epoch++) {
		cudaEvent_t start, stop;
		CUDA_CHECK_RETURN( cudaEventCreate(&start) );
		CUDA_CHECK_RETURN( cudaEventCreate(&stop) );
		CUDA_CHECK_RETURN( cudaEventRecord(start) );
		backpropagate(arguments.data_size, arguments.learning_rate, device);
		feedforward(arguments.data_size, device);
		auto acc_loss = evaluate(arguments.data_size, host, device);
		CUDA_CHECK_RETURN( cudaEventRecord(stop) );
		CUDA_CHECK_RETURN( cudaEventSynchronize(stop) );
		float accuracy = acc_loss.first, loss = acc_loss.second;
		average_accuracy += accuracy;
		float time = 0;
		CUDA_CHECK_RETURN( cudaEventElapsedTime(&time, start, stop) );
		fprintf(stderr, "epoch: %3u, accuracy: %f, loss: %f, time: %f ms\n", epoch, accuracy, loss, time);
		if (loss < arguments.epsilon) {
			break;
		}
	}
	average_accuracy /= epoch;
	FILE * output_file = fopen("results.txt", "w");
	fprintf(output_file, "%f\n", average_accuracy);
	fclose(output_file);
}

int main(int argc, char *argv[]) {
	Arguments arguments;
	parseArguments(argc, argv, arguments);

	HostStructures host;
	DeviceStructures device;

	FILE * input_file;
	if (!arguments.training_data_path.compare("-")) {
		input_file = stdin;
	}
	else {
		input_file = fopen(arguments.training_data_path.c_str(), "r");
	}
	CHECK_READ( fscanf(input_file, "%u", &arguments.data_size) );

	CUDA_CHECK_RETURN( cudaMallocHost(&host.labels, arguments.data_size * sizeof(int)) );
	CUDA_CHECK_RETURN( cudaMallocHost(&host.input_layer, arguments.data_size * layer_sizes[0] * sizeof(float)) );
	CUDA_CHECK_RETURN( cudaMallocHost(&host.evaluate_helper, grid_dim_1D * sizeof(float)) );
	for (int i = 0; i < layer_num - 1; i++) {
		CUDA_CHECK_RETURN( cudaMalloc(&device.weights[i], layer_sizes[i] * layer_sizes[i + 1] * sizeof(float)) );
		CUDA_CHECK_RETURN( cudaMalloc(&device.biases[i], layer_sizes[i + 1] * sizeof(float)) );
		CUDA_CHECK_RETURN( cudaMalloc(&device.layers_outputs[i], arguments.data_size * layer_sizes[i] * sizeof(float)) );
		CUDA_CHECK_RETURN( cudaMalloc(&device.errors[i], arguments.data_size * layer_sizes[i + 1] * sizeof(float)) );
		CUDA_CHECK_RETURN( cudaMalloc(&device.weights_partial_derivatives[i], layer_sizes[i] * layer_sizes[i + 1] * sizeof(float)) );
		CUDA_CHECK_RETURN( cudaMalloc(&device.weights_gradients[i], layer_sizes[i] * layer_sizes[i + 1] * sizeof(float)) );
		CUDA_CHECK_RETURN( cudaMalloc(&device.biases_gradients[i], layer_sizes[i + 1] * sizeof(float)) );
	}
	CUDA_CHECK_RETURN( cudaMalloc(&device.layers_outputs[layer_num - 1], arguments.data_size * layer_sizes[layer_num - 1] * sizeof(float)) );
	CUDA_CHECK_RETURN( cudaMalloc(&device.evaluate_helper, grid_dim_1D * sizeof(float)) );
	CUDA_CHECK_RETURN( cudaMalloc(&device.correct_predictions, sizeof(int)) );

	// weight initialization
	if (arguments.random) {
		curandGenerator_t gen;
		CURAND_CALL( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
		CURAND_CALL( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
		for (int i = 0; i < layer_num - 1; i++) {
			CURAND_CALL( curandGenerateUniform(gen, device.weights[i], layer_sizes[i] * layer_sizes[i + 1]) );
		}
		for (int i = 0; i < layer_num - 1; i++) {
			CURAND_CALL( curandGenerateUniform(gen, device.biases[i], layer_sizes[i + 1]) );
		}
	}
	else {
		for (int i = 0; i < layer_num - 1; i++) {
			CUDA_CHECK_RETURN( cudaMemset(device.weights[i], initial_weight, layer_sizes[i] * layer_sizes[i + 1] * sizeof(float)) );
		}
		for (int i = 0; i < layer_num - 1; i++) {
			CUDA_CHECK_RETURN( cudaMemset(device.biases[i], initial_weight, layer_sizes[i + 1] * sizeof(float)) );
		}
	}

	// input
	for (int i = 0; i < arguments.data_size; i++) {
		for (int j = 0; j < feature_num; j++) {
			CHECK_READ( fscanf(input_file, "%f", &host.input_layer[i * feature_num + j]) );
		}
		CHECK_READ( fscanf(input_file, "%u", &host.labels[i]) );
	}
	fclose(input_file);

	// data normalization
	for (int i = 0; i < arguments.data_size; i++) {
		for (int j = 0; j < feature_num; j++) {
			host.input_layer[i * feature_num + j] /= 255.0f;
		}
	}

	CUDA_CHECK_RETURN( cudaMemcpy(device.layers_outputs[0], host.input_layer, arguments.data_size * layer_sizes[0] * sizeof(float), cudaMemcpyHostToDevice) );

#if const_labels_size > 0
	init_const_labels(arguments.data_size, host.labels);
#else
	CUDA_CHECK_RETURN( cudaMalloc(&device.labels, arguments.data_size * sizeof(int)) );
	CUDA_CHECK_RETURN( cudaMemcpy(device.labels, host.labels, arguments.data_size * sizeof(int), cudaMemcpyHostToDevice) );
#endif

	CUDA_CHECK_KERNEL
	train(host, device, arguments);

	CUDA_CHECK_RETURN( cudaThreadExit() );

	return 0;
}

