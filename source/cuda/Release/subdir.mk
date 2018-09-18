################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../kernels.cu \
../mlp.cu 

OBJS += \
./kernels.o \
./mlp.o 

CU_DEPS += \
./kernels.d \
./mlp.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3 -std=c++11 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -m64 -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


