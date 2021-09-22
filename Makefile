CC = gcc
NVCC=nvcc
NVCCFLAG = -std=c++11 -c -arch=sm_52

.PHONY : dummy main.x clean

dummy:
	@echo make cuda compile all CUDA sources

cuda:
	main.x

main.x:	main.cu
		$(NVCC) $(NVCCFLAG) $< -o output

clean:
	rm -f *.x *.mod *.ptx