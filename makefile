all: cpu

cpu: src/cpu_backend.c
	gcc -Wall -lm -c src/cpu_backend.c -o bin/cpu_backend.so