#include <cfloat>
#include <cassert>
#include "base.cuh"
#include "kernels.cuh"
#include "debug.cuh"


#if const_labels_size > 0
__constant__ int const_labels[const_labels_size];
void init_const_labels(int size, int *labels) {
	assert(size <= const_labels_size);
	CUDA_CHECK_RETURN( cudaMemcpyToSymbol(const_labels, labels, size * sizeof(int)) );
}
#endif

__device__ __forceinline__ float relu(float x) {
	return x * (x > 0);
}

__device__ __forceinline__ float relu_prime(float x) {
	return (x > 0);
}

__device__ __forceinline__ int align(int x) {
	return ((x + block_size - 1) / block_size) * block_size;
}

__global__ void compute_hidden_layer_outputs(
		int prev_layer_size, int curr_layer_size, int data_size,
		float *prev_outputs, float *curr_outputs, float *weights, float *biases) {
	__shared__ float prev_outputs_sub[block_size][block_size];
	__shared__ float weights_sub[block_size][block_size];
	__shared__ float biases_sub[block_size];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	dim3 offset = dim3(blockIdx.x * block_size, blockIdx.y * block_size);
	dim3 stride = dim3(block_size * gridDim.x, block_size * gridDim.y);
	dim3 bound = dim3(align(curr_layer_size), align(data_size));
	for (int iy = offset.y + ty; iy < bound.y; iy += stride.y) {
		for (int ix = offset.x + tx; ix < bound.x; ix += stride.x) {
			float activation = 0;
			if (ty == 0) {
				biases_sub[tx] = ix < curr_layer_size ? biases[ix] : 0;
			}
			for (int j = 0; j < prev_layer_size; j += block_size) {
				int jx = j + tx;
				int jy = j + ty;
				prev_outputs_sub[ty][tx] =
					iy < data_size && jx < prev_layer_size ?
					prev_outputs[iy * prev_layer_size + jx] : 0;
				weights_sub[ty][tx] =
					jy < prev_layer_size && ix < curr_layer_size ?
					weights[jy * curr_layer_size + ix] : 0;
				__syncthreads();
#pragma unroll
				for (int k = 0; k < block_size; k++) {
					activation += prev_outputs_sub[ty][k] * weights_sub[k][tx];
				}
				__syncthreads();
			}
			if (iy < data_size && ix < curr_layer_size) {
				curr_outputs[iy * curr_layer_size + ix] = relu(activation + biases_sub[tx]);
			}
		}
	}
}

__global__ void compute_softmax_layer_activations(
		int prev_layer_size, int curr_layer_size, int data_size,
		float *prev_outputs, float *curr_outputs, float *weights, float *biases) {
	__shared__ float prev_outputs_sub[block_size][block_size];
	__shared__ float weights_sub[block_size][block_size];
	__shared__ float biases_sub[block_size];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	dim3 offset = dim3(blockIdx.x * block_size, blockIdx.y * block_size);
	dim3 stride = dim3(block_size * gridDim.x, block_size * gridDim.y);
	dim3 bound = dim3(align(curr_layer_size), align(data_size));
	for (int iy = offset.y + ty; iy < bound.y; iy += stride.y) {
		for (int ix = offset.x + tx; ix < bound.x; ix += stride.x) {
			float activation = 0;
			if (ty == 0) {
				biases_sub[tx] = ix < curr_layer_size ? biases[ix] : 0;
			}
			for (int j = 0; j < prev_layer_size; j += block_size) {
				int jx = j + tx;
				int jy = j + ty;
				prev_outputs_sub[ty][tx] =
					iy < data_size && jx < prev_layer_size ?
					prev_outputs[iy * prev_layer_size + jx] : 0;
				weights_sub[ty][tx] =
					jy < prev_layer_size && ix < curr_layer_size ?
					weights[jy * curr_layer_size + ix] : 0;
				__syncthreads();
#pragma unroll
				for (int k = 0; k < block_size; k++) {
					activation += prev_outputs_sub[ty][k] * weights_sub[k][tx];
				}
				__syncthreads();
			}
			if (iy < data_size && ix < curr_layer_size) {
				curr_outputs[iy * curr_layer_size + ix] = activation + biases_sub[tx];
			}
		}
	}
}

// Compute max in every row, sum elements in every row
// Divide every element in row by sum of normalised exponents for every row
__global__ void compute_softmax_layer_outputs(
		int layer_size, int data_size, float *layer_outputs) {
	int tx = threadIdx.x;
	int stride = block_size * gridDim.x;
	int offset = blockIdx.x * block_size;
	for (int ix = offset + tx; ix < data_size; ix += stride) {
		float max_activation = -FLT_MAX;
		float sum_exps = 0;
#pragma unroll
		for (int jx = 0; jx < layer_size; jx++) {
			max_activation = fmaxf(max_activation, layer_outputs[ix * layer_size + jx]);
		}
#pragma unroll
		for (int jx = 0; jx < layer_size; jx++) {
			sum_exps += expf(layer_outputs[ix * layer_size + jx] - max_activation);
		}
#pragma unroll
		for (int jx = 0; jx < layer_size; jx++) {
			layer_outputs[ix * layer_size + jx] =
					expf(layer_outputs[ix * layer_size + jx] - max_activation) / sum_exps;
		}
	}
}

__global__ void compute_output_layer_errors(
		int layer_size, int data_size, int *labels, float *layer_outputs, float *errors) {
	int tx = threadIdx.x;
	int stride = block_size * gridDim.x;
	int offset = blockIdx.x * block_size;
	for (int ix = offset + tx; ix < data_size; ix += stride) {
		for (int jx = 0; jx < layer_size; jx++) {
#if const_labels_size > 0
			errors[ix * layer_size + jx] = layer_outputs[ix * layer_size + jx] - (const_labels[ix] == jx);
#else
			errors[ix * layer_size + jx] = layer_outputs[ix * layer_size + jx] - (labels[ix] == jx);
#endif

		}
	}
}

__global__ void compute_hidden_layer_errors(
		int next_layer_size, int curr_layer_size, int data_size,
		float *outputs, float *weights, float *next_errors, float *curr_errors) {
	__shared__ float next_errors_sub[block_size][block_size];
	__shared__ float weights_sub[block_size][block_size];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	dim3 offset = dim3(blockIdx.x * block_size, blockIdx.y * block_size);
	dim3 stride = dim3(block_size * gridDim.x, block_size * gridDim.y);
	dim3 bound = dim3(align(curr_layer_size), align(data_size));
	for (int iy = offset.y + ty; iy < bound.y; iy += stride.y) {
		for (int ix = offset.x + tx; ix < bound.x; ix += stride.x) {
			float weighted_error = 0;
			for (int j = 0; j < next_layer_size; j += block_size) {
				int jx = j + tx;
				int jy = j + ty;
				next_errors_sub[ty][tx] =
					iy < data_size && jx < next_layer_size ?
					next_errors[iy * next_layer_size + jx] : 0;
				// weights matrix transposition
				weights_sub[ty][tx] =
					ix < curr_layer_size && jy < next_layer_size ?
					weights[ix * next_layer_size + jy] : 0;
				__syncthreads();
#pragma unroll
				for (int k = 0; k < block_size; k++) {
					weighted_error += next_errors_sub[ty][k] * weights_sub[k][tx];
				}
				__syncthreads();
			}
			if (iy < data_size && ix < curr_layer_size) {
				curr_errors[iy * curr_layer_size + ix] =
					relu_prime(outputs[iy * curr_layer_size + ix]) * weighted_error;
			}
		}
	}
}

__global__ void compute_weights_gradients(
		int prev_layer_size, int curr_layer_size, int data_size,
		float *outputs, float *errors, float *weights_gradients) {
	__shared__ float outputs_sub[block_size][block_size];
	__shared__ float errors_sub[block_size][block_size];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	dim3 offset = dim3(blockIdx.x * block_size, blockIdx.y * block_size);
	dim3 stride = dim3(block_size * gridDim.x, block_size * gridDim.y);
	dim3 bound = dim3(align(curr_layer_size), align(prev_layer_size));
	for (int iy = offset.y + ty; iy < bound.y; iy += stride.y) {
		for (int ix = offset.x + tx; ix < bound.x; ix += stride.x) {
			float gradient = 0;
			for (int j = 0; j < data_size; j += block_size) {
				int jx = j + tx;
				int jy = j + ty;
				// layer outputs matrix transposition
				outputs_sub[ty][tx] =
					jx < data_size && iy < prev_layer_size ?
					outputs[jx * prev_layer_size + iy] : 0;
				errors_sub[ty][tx] =
					jy < data_size && ix < curr_layer_size ?
					errors[jy * curr_layer_size + ix] : 0;
				__syncthreads();
#pragma unroll
				for (int k = 0; k < block_size; k++) {
					gradient += outputs_sub[ty][k] * errors_sub[k][tx];
				}
				__syncthreads();
			}
			if (iy < prev_layer_size && ix < curr_layer_size) {
				weights_gradients[iy * curr_layer_size + ix] = gradient / data_size;
			}
		}
	}
}

__global__ void compute_biases_gradients(
		int layer_size, int data_size, float *errors, float *biases_gradients) {
	__shared__ float errors_sub[block_dim_1D];
	int tx = threadIdx.x;
	const int bx = blockIdx.x;
	int stride = gridDim.x;
	int bound = data_size - (data_size % block_size);
	if (tx == 0) {
		for (int ix = bx; ix < layer_size; ix += stride) {
			biases_gradients[ix] = 0;
		}
	}
	for (int ix = bx; ix < layer_size; ix += stride) {
		for (int jx = tx; jx < bound; jx += block_size) {
			errors_sub[tx] = errors[jx * layer_size + ix];
			__syncthreads();
			for (int k = block_size / 2; k != 0; k /= 2) {
				if (tx < k) {
					errors_sub[tx] += errors_sub[tx + k];
				}
				__syncthreads();
			}
			if (tx == 0) {
				biases_gradients[ix] += errors_sub[0];
			}
		}
		if (tx == 0) {
			float err = 0;
			for (int jx = bound; jx < data_size; jx++) {
				err += errors[jx * layer_size + ix];
			}
			biases_gradients[ix] += err;
		}
	}
}

__global__ void update_weights(
		float learning_rate, int prev_layer_size, int curr_layer_size,
		float *weights_gradients, float *weights) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	dim3 offset = dim3(blockIdx.x * block_size, blockIdx.y * block_size);
	dim3 stride = dim3(block_size * gridDim.x, block_size * gridDim.y);
	for (int iy = offset.y + ty; iy < prev_layer_size; iy += stride.y) {
		for (int ix = offset.x + tx; ix < curr_layer_size; ix += stride.x) {
			weights[iy * curr_layer_size + ix] -= learning_rate * weights_gradients[iy * curr_layer_size + ix];
		}
	}
}

__global__ void update_biases(
		float learning_rate, int layer_size, float *biases_gradients, float *biases) {
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int stride = block_size * gridDim.x;
	for (int ix = bx * block_size + tx; ix < layer_size; ix += stride) {
		biases[ix] -= learning_rate * biases_gradients[ix];
	}
}

__global__ void evaluate_loss(
		int layer_size, int data_size, int *labels, float *outputs, float *evaluate_helper) {
	__shared__ float losses[block_dim_1D];
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int boundary = (layer_size * data_size) - ((layer_size * data_size) % block_size);
	float eps = 0.000001;
	if (tx == 0) {
		evaluate_helper[bx] = 0;
	}
	for (int ix = bx * blockDim.x + tx; ix < boundary; ix += blockDim.x * gridDim.x) {
#if const_labels_size > 0
		int prediction = (const_labels[ix / layer_size] == ix % layer_size);
#else
		int prediction = (labels[ix / layer_size] == ix % layer_size);
#endif
		losses[tx] = prediction * logf(outputs[ix] + eps) + (1 - prediction) * logf(1 - outputs[ix] + eps);
		__syncthreads();
		for (int j = blockDim.x / 2; j != 0; j /= 2) {
			if (tx < j) {
				losses[tx] += losses[tx + j];
			}
			__syncthreads();
		}
		if (tx == 0) {
			evaluate_helper[bx] -= losses[0] / data_size;
		}
	}
	if (tx == 0) {
		float end_losses = 0;
		for (int ix = boundary + bx; ix < layer_size * data_size; ix += gridDim.x) {
#if const_labels_size > 0
			int prediction = (const_labels[ix / layer_size] == ix % layer_size);
#else
			int prediction = (labels[ix / layer_size] == ix % layer_size);
#endif
			end_losses += prediction * logf(outputs[ix] + eps) + (1 - prediction) * logf(1 - outputs[ix] + eps);
		}
		evaluate_helper[bx] -= end_losses / data_size;
	}
}

__global__ void evaluate_accuracy(
		int layer_size, int data_size,
		int *labels, float *outputs, int *correct_predictions) {
	__shared__ int correct_predictions_block;
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int correct_predictions_thread = 0;
	if (tx == 0) {
		if (bx == 0) {
			*correct_predictions = 0;
		}
		correct_predictions_block = 0;
	}
	for (int iy = bx * blockDim.x + tx; iy < data_size; iy += blockDim.x * gridDim.x) {
		int max_index = 0;
		float max_prediction = -FLT_MAX;
		for (int ix = 0; ix < layer_size; ix++) {
			if (max_prediction < outputs[iy * layer_size + ix]) {
				max_prediction = outputs[iy * layer_size + ix];
				max_index = ix;
			}
		}
#if const_labels_size > 0
		correct_predictions_thread += (max_index == const_labels[iy]);
#else
		correct_predictions_thread += (max_index == labels[iy]);
#endif
	}
	atomicAdd(&correct_predictions_block, correct_predictions_thread);
	if (tx == 0) {
		atomicAdd(correct_predictions, correct_predictions_block);
	}
}

//__global__ void has_nan(int size, float *arr) {
//	for (int i = 0; i < size; i++) {
//		if (!(::isfinite(arr[i]))) {
//			printf("INVALID VALUE: %i: %f\n", i, arr[i]);
//			return;
//		}
//	}
//}

//__global__ void lookup(float *x) { *x += 0.000000; }
