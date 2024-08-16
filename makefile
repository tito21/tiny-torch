all: cpu

cpu: cpu_backend.c
	gcc -Wall -c cpu_backend.c -o bin/cpu_backend.so