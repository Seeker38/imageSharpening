

cmake_minimum_required(VERSION 3.0.0)
project(imageSharpening VERSION 0.1.0 LANGUAGES C CXX CUDA)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(CUDA REQUIRED)
enable_language(CUDA)

# cmake_policy(SET CMP0104 NEW)  # Set the policy to address the warning


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--std=c++17;)    

add_executable(imageSharpening  
    cpu.cpp)

target_link_libraries(imageSharpening ${OpenCV_LIBS} )

set_target_properties(imageSharpening PROPERTIES CUDA_ARCHITECTURES "native") # Set the CUDA_ARCHITECTURES appropriately


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)