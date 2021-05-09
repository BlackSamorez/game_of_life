#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>


#define gpuErrchk(ans) { gpuAssert( (ans), __FILE__, __LINE__ ); }

inline void
gpuAssert( cudaError_t code, const char * file, int line, bool abort = true )
{
	if ( cudaSuccess != code )
	{
		fprintf( stderr, "\nGPUassert: %s %s %d\n", cudaGetErrorString( code ), file, line );
		if ( abort )
			exit( code );
	}


	return;

}

namespace cuda_kernels{
	__host__ __device__ int index_from_coordinates(int column, int row, int level, int length, int width, int heighth){
		return (column % length) + (row % width) * length + (level % heighth) * length * width;
	}
	__host__ __device__ int calc_neighbours(int* grid, int column, int row, int level, int length, int width, int heighth){
		return grid[index_from_coordinates(column + -1, row + -1, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + -1, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + -1, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + 0, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + 0, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + 0, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + 1, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + 1, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + 1, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + -1, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + -1, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + -1, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + 0, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + 0, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + 1, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + 1, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + 1, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + -1, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + -1, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + -1, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + 0, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + 0, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + 0, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + 1, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + 1, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + 1, level + 1, length, width, heighth)];
	}

	__global__ void calc_next_generation_all_global(int* current_grid, int* next_grid, int length, int width, int heighth){ 
		// Call to global memory approx 28 times per cell - very slow
		int column = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int level = blockIdx.z * blockDim.z + threadIdx.z;

		int place = index_from_coordinates(column, row, level, length, width, heighth);

		if (column < length and row < width and level < heighth)
		{
			int state = current_grid[place]; // slow
			int neighbours = calc_neighbours(current_grid, column, row, level, length, width, heighth); // 26 * slow

			next_grid[place] = 0; // slow
			if (state == 0 and (neighbours == 6 or neighbours == 7))
			{
				next_grid[place] = 1; // slow
			}
			if (state == 1 and (neighbours == 4 or neighbours == 5 or neighbours == 6 or neighbours == 7))
			{
				next_grid[place] = 1; // slow
			}
		}
	}

	__global__ void calc_next_generation_shared_areas(int* current_grid, int* next_grid, int length, int width, int heighth){
		// Call to global memory approx 2 times per cell - fast
		//copy areas for each block into it's shared memory and then calculate neighbours
		int column = blockIdx.x * blockDim.x + threadIdx.x - 1;
		int row = blockIdx.y * blockDim.y + threadIdx.y - 1;
		int level = blockIdx.z * blockDim.z + threadIdx.z - 1;

		if (column < length + 2 and row < width + 2 and level < heighth + 2)
		{
			int place = index_from_coordinates(column, row, level, length, width, heighth);

			__shared__ int area[8192]; // 32kb of shared memory

			area[threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y] = current_grid[place]; // slow

			__syncthreads();

			// now we can calculate neighbours fast in a slightly smaller area
			if ((threadIdx.x > 0 and threadIdx.x < blockDim.x - 1 and threadIdx.y > 0 and threadIdx.y < blockDim.y - 1 and threadIdx.z > 0 and threadIdx.z < blockDim.z - 1) or column > length or row > width or level > heighth)
			{
				int neighbours;
				neighbours = calc_neighbours(area, threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z); // fast

				int state = area[threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y];
				
				int newstate = 0;
				if (state == 0 and (neighbours == 6 or neighbours == 7))
				{
					newstate = 1;
				}
				if (state == 1 and (neighbours == 4 or neighbours == 5 or neighbours == 6 or neighbours == 7))
				{
					newstate = 1;
				}

				next_grid[place] = newstate; //slow
			}
		}
	}
}

void read_size(std::string filename, int* length, int* width, int* heighth){
	std::ifstream data(filename);
	std::string line;
	std::getline(data, line);

	std::stringstream lineStream(line);
	std::vector<std::string> parsedRow;
	std::string cell;

	while(std::getline(lineStream, cell, ' '))
    {
        parsedRow.push_back(cell);
    }

    *length = std::stoi(parsedRow[0]);
    *width = std::stoi(parsedRow[1]);
    *heighth = std::stoi(parsedRow[2]);
}

void read_input(std::string filename, int* state){
	std::ifstream data(filename);
	std::string line;
	std::getline(data, line);

	std::stringstream lineStream(line);
	std::vector<std::string> parsedRow;
	std::string cell;

	while(std::getline(lineStream, cell, ' '))
    {
        parsedRow.push_back(cell);
    }

    int length = std::stoi(parsedRow[0]);
    int width = std::stoi(parsedRow[1]);
    int heighth = std::stoi(parsedRow[2]);

    while(std::getline(data,line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> parsedRow;
        while(std::getline(lineStream, cell ,' '))
        {
            parsedRow.push_back(cell);
        }
        if (std::stoi(parsedRow[0]) != 0 or std::stoi(parsedRow[1]) != 0 or std::stoi(parsedRow[2]) != 0 or std::stoi(parsedRow[3]) != 0)
        {
        	state[std::stoi(parsedRow[1]) + std::stoi(parsedRow[2]) * length + std::stoi(parsedRow[3]) * length * width] = std::stoi(parsedRow[0]);
        }
    }
}

void append_state_to_file(std::string filename, int* field_d, int length, int width, int heighth){
	int field_h[length * width * heighth];
	size_t size = length * width * heighth * sizeof(int);
	cudaMemcpy(field_h, field_d, size, cudaMemcpyDeviceToHost);

	std::ofstream file;
	file.open(filename, std::ios_base::app);

	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			for (int k = 0; k < heighth; ++k)
			{
				int place = i + j * length + k * length * width;
				if (field_h[place] != 0){
					file << field_h[place] << " " << i << " " << j << " " << k << "\n"; 
				}
			}
		}
	}
	file << "0 0 0 0\n";
}

int main(){
	std::string filename;
	std::cin >> filename;
	int length = 20, width = 20, heighth = 20;

	read_size(filename, &length, &width, &heighth);

	int field_h[length * width * heighth];

	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			for (int k = 0; k < heighth; ++k)
			{
				field_h[i + j * length + k * length * width] = 0;
			}
		}
	}

	read_input(filename, field_h);

	int* field0_d;
	int* field1_d;
	size_t size = length * width * heighth * sizeof(int);
	gpuErrchk(cudaMalloc(&field0_d, size));
	gpuErrchk(cudaMalloc(&field1_d, size));

	gpuErrchk(cudaMemcpy(field0_d, field_h, size, cudaMemcpyHostToDevice));

	std::ofstream ofs;
	ofs.open("test.out", std::ofstream::out | std::ofstream::trunc);
	ofs.close();

	dim3 tpb(10, 10, 10);
	dim3 bpg(length / tpb.x + 1, width / tpb.y + 1, heighth / tpb.z + 1);



	for (int i = 0; i < 10; ++i)
	{
		append_state_to_file("test.out", field0_d, length, width, heighth);
		cuda_kernels::calc_next_generation_all_global<<<bpg, tpb>>>(field0_d, field1_d, length, width, heighth);
		gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

		std::swap(field0_d, field1_d);
	}

return 0;
}