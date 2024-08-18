
all: cpu cuda memutils

cpu: src/cpu_backend_implementation.c
	g++ -shared -lm -o build/cpu_backend.so src/cpu_backend_implementation.c

cuda: src/cuda_backend_implementation.cu
	nvcc -shared --compiler-options -fPIC -o build/cuda_backend.so src/cuda_backend_implementation.cu

memutils: src/memutils.c
	g++ -shared  src/memutils.c -o build/memutils.so

