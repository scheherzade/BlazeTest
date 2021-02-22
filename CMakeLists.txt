cmake_minimum_required(VERSION 3.8)
project(BlazeTest CUDA CXX)
find_package(CUDA)

#find_package(OpenMP)
#if (OPENMP_FOUND)
#    message("OPENMP FOUND")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

#find_package (Threads)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lblas -L/usr/lib64/ -llapack")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -llapack -L)

find_package( LAPACK REQUIRED )
include_directories(${LAPACK_INCLUDE_DIR})
message ("lapack" ${LAPACK_INCLUDE_DIR})

find_package( BLAS REQUIRED )
include_directories(${BLAS_INCLUDE_DIR})

#include_directories(~/src/OpenBLAS/)
#target_link_libraries(BlazeTest "/home/shahrzad/src/OpenBLAS_clang/OpenBLAS/libopenblas_haswellp-r0.3.1.dev.a")

#add_library(libopenblas SHARED IMPORTED)
#set_target_properties(libopenblas PROPERTIES IMPORTED_LOCATION "/home/shahrzad/src/OpenBLAS/libopenblas.so")

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")



#find_package(blaze REQUIRED)

add_definitions(-DBLAZE_USE_HPX_THREADS)
#add_definitions(-DNEW_BACKEND)

#
include_directories(/home/shahrzad/lib/blaze/include/)
include_directories(/home/shahrzad/src/blaze_cuda/)
include_directories(/home/shahrzad/src/thrust/)
include_directories(${CUDA_INCLUDE_DIRS})

#add_executable(BlazeTest main.cpp)

find_package(HPX REQUIRED)
#include_directories(${HPX_INCLUDE_DIRS})
#link_directories(${HPX_LIBRARY_DIR})
#hpx_setup_target(BlazeTest)
add_hpx_executable(BlazeTest SOURCES main.cu)
#set_target_properties(
#        BlazeTest
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(BlazeTest PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${CUDA_LIBRARIES})
#


get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()




