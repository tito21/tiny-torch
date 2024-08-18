
all: cpu cuda memutils

cpu: src/cpu_backend_implementation.c src/cpu_backend_implementation.h
	g++ -shared -lm -o build/cpu_backend.so src/cpu_backend_implementation.c

cuda: src/cuda_backend_implementation.cu src/cuda_backend_implementation.cuh
	nvcc -shared --compiler-options -fPIC -o build/cuda_backend.so src/cuda_backend_implementation.cu

memutils: src/memutils.c src/memutils.h
	g++ -shared  src/memutils.c -o build/memutils.so

clean:
	rm build/cpu_backend.so build/cuda_backend.so build/memutils.so