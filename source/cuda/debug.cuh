#ifndef DEBUG_CUH_
#define DEBUG_CUH_

#include <iostream>


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static void CheckCudaKernelAux (const char *, unsigned);
#define CUDA_CHECK_KERNEL CheckCudaKernelAux(__FILE__,__LINE__);

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "(" << err << ") at "<< file << ":" << line << std::endl;
	exit(1);
}

static void CheckCudaKernelAux(const char *file, unsigned line) {
	cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess) {
		std::cerr << cudaGetErrorString(errSync) << "("<<errSync<< ") at "<< file << ":" << line << std::endl;
		exit(1);
	}
	if (errAsync != cudaSuccess) {
		std::cerr << cudaGetErrorString(errAsync) << "("<< errAsync << ") at "<< file << ":" << line << std::endl;
		exit(1);
	}
}

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

static void check_read_value(const char * file, unsigned line, int len) {
	if (len <= 0) {
		std::cerr <<"No input value at "<< file << ":" << line << std::endl;
		exit(1);
	}
}
#define CHECK_READ(len) check_read_value(__FILE__,__LINE__, len);


#endif /* DEBUG_CUH_ */
