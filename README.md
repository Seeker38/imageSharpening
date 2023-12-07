# Image Sharpening

Project that sharpen images using cuda and c++



## Building project

Ensure that you have c++, nvcc and cuda libraries installed.

Requirement :
1. opencv 4.8.0
2. microsoft visual studio 2022
3. CMake 3.26.6
4. CUDA Toolkit 12.1

Installation :
1. First check the Requirements and make sure vscode has been installed all the estensions involving c/c++, cmake, code runner,..
2. Delete build folder
3. Press Ctrl, shift + p and select cmake INIT project, do the typing.
4. In CMakeLists.txt, make sure the correct folder name is being filled.
5. In CMakeLists.txt, in add_executable,you can change which one to run, just make sure do not let GPU and CPU uncomment at the same time.
6. Press Launch and enjoy. 


File (make sure you change the input and output path):
1. Run code by GPU --> gpu.cu
2. Run code by CPu --> cpu.cpp
3. Run images comparison --> mainCompare.cpp 

CMake the project using Cmake accorfing to your c/c++ version and CUDA_ARCHITECTURES

```bash
set_property(TARGET my_target PROPERTY "${CUDA_ARCH_LIST}")
```


## Sources
The code is based on 
[https://www.researchgate.net/publication/307695773_Optimizing_Image_Sharpening_Algorithm_on_GPU](https://www.researchgate.net/publication/307695773_Optimizing_Image_Sharpening_Algorithm_on_GPU)

