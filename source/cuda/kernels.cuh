/*
 * kernels.cuh
 *
 *  Created on: 5 maj 2018
 *      Author: jan
 */

#ifndef KERNELS_CUH_
#define KERNELS_CUH_


void init_const_labels(int size, int *labels);

__global__ void compute_hidden_layer_outputs(
		int prev_layer_size, int curr_layer_size, int data_size,
		float *prev_outputs, float *curr_outputs, float *weights, float *biases);

__global__ void compute_softmax_layer_activations(
		int prev_layer_size, int curr_layer_size, int data_size,
		float *prev_outputs, float *curr_outputs, float *weights, float *biases);

__global__ void compute_softmax_layer_outputs(
		int layer_size, int data_size, float *layer_outputs);

__global__ void compute_output_layer_errors(
		int layer_size, int data_size, int *labels, float *layer_outputs, float *errors);

__global__ void compute_hidden_layer_errors(
		int next_layer_size, int curr_layer_size, int data_size,
		float *outputs, float *weights, float *next_errors, float *curr_errors);

__global__ void compute_weights_gradients(
		int prev_layer_size, int curr_layer_size, int data_size,
		float *outputs, float *errors, float *weights_gradients);

__global__ void compute_biases_gradients(
		int layer_size, int data_size, float *errors, float *biases_gradients);

__global__ void update_weights(
		float learning_rate, int prev_layer_size, int curr_layer_size,
		float *weights_gradients, float *weights);

__global__ void update_biases(
		float learning_rate, int layer_size, float *biases_gradients, float *biases);

__global__ void evaluate_loss(
		int layer_size, int data_size, int *labels, float *outputs, float *evaluate_helper);

__global__ void evaluate_accuracy(
		int layer_size, int data_size,
		int *labels, float *outputs, int *correct_predictions);


#endif /* KERNELS_CUH_ */
