#include <iostream>

// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

using namespace std;

__host__ __device__ int sum(dim3 three){
	return three.x + three.y + three.z;
}

__global__ void three_to_two(int* data_3d, int* data_2d, int size){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < size && y < size){
		for (int i = 0; i < size; ++i)
		{
			data_2d[x + size * y] += data_3d[x + size * y + size * size * i];
		}
	}
	
}

__global__ void two_to_one(int* data_2d, int* data_1d, int size){
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < size){
		for (int i = 0; i < size; ++i)
		{
			data_1d[x] += data_2d[x + size * i];
		}
	}
}

__global__ void one_to_zero(int* data_1d, int* data_0d, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < 1){
		for (int i = 0; i < size; ++i)
		{
			data_0d[0] += data_1d[i];
		}
	}
}

__global__ void test(dim3 data){
	return;
}


int main(){
	int* data_3d;
	int* data_2d;
	int* data_1d;
	int* data_0d;
	cudaMalloc(&data_3d, 216 * sizeof(int));
	cudaMalloc(&data_2d, 36 * sizeof(int));
	cudaMalloc(&data_1d, 6 * sizeof(int));
	cudaMalloc(&data_0d, 1 * sizeof(int));

	int* data_3h = new int(216);
	for (int i = 0; i < 216; ++i)
	{
		data_3h[i] = i;
	}

	cudaMemcpy(data_3d, data_3h, 216 * sizeof(int), cudaMemcpyHostToDevice);

	dim3 tpb(3, 3);
	dim3 bpg(2, 2);

	three_to_two<<<bpg, tpb>>>(data_3d, data_2d, 6);

	tpb = (3);
	bpg = (2);

	two_to_one<<<bpg, tpb>>>(data_2d, data_1d, 6);

	tpb = (1);
	bpg = (1);

	one_to_zero<<<bpg, tpb>>>(data_1d, data_0d, 6);

	int answer;

	cudaMemcpy(&answer, data_0d, sizeof(int), cudaMemcpyDeviceToHost);

	std::cout << answer << "\n";

	test<<<bpg, tpb>>>(bpg);

	std::cout << "done\n";

return 0;
}