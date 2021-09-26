NVCC=nvcc
NVCCFLAG =  -gencode=arch=compute_52,code=\"sm_52,compute_52\"
W10KIT = "E:\Windows Kits\10\Include\10.0.19041.0\ucrt"
MSVS = "E:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\include"
CUDAINCLUDE = "%CUDA_PATH%\include"
OBJFLAG = -G -maxrregcount=0  --machine 64 --compile -cudart static -g -DWIN32 -DWIN64 -D_DEBUG -D_CONSOLE -D_MBCS --keep --keep-dir "Release" -Xcompiler "/EHsc /W3 /nologo /Od /FS /Zi /RTC1 /MDd "
CUDACRT = "%CUDA_PATH%\bin/crt"
CUDALIB = "%CUDA_PATH%\lib\x64"
LIB = cudart_static.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib cudart.lib cudadevrt.lib
INPUTOBJ = "Release\main.cu.obj" "Release\tester.cu.obj" "Release\utils.cu.obj" "Release\inputgenerator.cu.obj" "Release\kmeans.cu.obj" "Release\dbscan.cu.obj" "Release\kmeansCUDA.cu.obj" "Release\dbscanCUDA.cu.obj"

all: dir main tester utils inputgenerator kmeans dbscan kmeansCUDA dbscanCUDA exe clean

dir:
	@echo off
	IF exist "Release" ( echo "Release" exists ) ELSE ( mkdir "Release" && echo "Release" created)

main: 
	$(NVCC) $(NVCCFLAG) --use-local-env -x cu -rdc=true -I$(W10KIT) -I$(MSVS) -I$(CUDAINCLUDE) $(OBJFLAG)  -o "Release\main.cu.obj" main.cu -Xcompiler /openmp

tester: 
	$(NVCC) $(NVCCFLAG) --use-local-env -x cu -rdc=true -I$(W10KIT) -I$(MSVS) -I$(CUDAINCLUDE) $(OBJFLAG)  -o "Release\tester.cu.obj" tester.cu -Xcompiler /openmp

utils: 
	$(NVCC) $(NVCCFLAG) --use-local-env -x cu -rdc=true -I$(W10KIT) -I$(MSVS) -I$(CUDAINCLUDE) $(OBJFLAG)  -o "Release\utils.cu.obj" utils.cu -Xcompiler /openmp

inputgenerator: 
	$(NVCC) $(NVCCFLAG) --use-local-env -x cu -rdc=true -I$(W10KIT) -I$(MSVS) -I$(CUDAINCLUDE) $(OBJFLAG)  -o "Release\inputgenerator.cu.obj" inputgenerator.cu -Xcompiler /openmp

kmeans: 
	$(NVCC) $(NVCCFLAG) --use-local-env -x cu -rdc=true -I$(W10KIT) -I$(MSVS) -I$(CUDAINCLUDE) $(OBJFLAG)  -o "Release\kmeans.cu.obj" kmeans.cu -Xcompiler /openmp

dbscan: 
	$(NVCC) $(NVCCFLAG) --use-local-env -x cu -rdc=true -I$(W10KIT) -I$(MSVS) -I$(CUDAINCLUDE) $(OBJFLAG)  -o "Release\dbscan.cu.obj" dbscan.cu -Xcompiler /openmp

kmeansCUDA: 
	$(NVCC) $(NVCCFLAG) --use-local-env -x cu -rdc=true -I$(W10KIT) -I$(MSVS) -I$(CUDAINCLUDE) $(OBJFLAG)  -o "Release\kmeansCUDA.cu.obj" kmeansCUDA.cu -Xcompiler /openmp

dbscanCUDA: 
	$(NVCC) $(NVCCFLAG) --use-local-env -x cu -rdc=true -I$(W10KIT) -I$(MSVS) -I$(CUDAINCLUDE) $(OBJFLAG)  -o "Release\dbscanCUDA.cu.obj" dbscanCUDA.cu -Xcompiler /openmp

exe:
	$(NVCC) -o "main.exe" -Xcompiler "/EHsc /W3 /nologo /Od  /Zi /RTC1 /MDd " -L$(CUDACRT) -L$(CUDALIB) $(LIB) $(NVCCFLAG) -G --machine 64 $(INPUTOBJ) -Xcompiler /openmp

clean: fileClean dirClean libClean

fileClean:
	erase /Q "Release"

dirClean:
	rmdir /Q "Release"

libClean:
	erase /Q *.pdb *.lib *.exp
