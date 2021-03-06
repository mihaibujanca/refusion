cmake_minimum_required(VERSION 2.8)
project(refusion)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
set(CMAKE_CXX_FLAGS "-std=c++11") # Eigen raises deprecated declarations warning.

set(CUDA_SEPARABLE_COMPILATION ON)

include_directories(
        src
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
)

add_library(eigen_wrapper SHARED src/tracker/eigen_wrapper.cpp)
cuda_add_library(refusion SHARED
        src/utils/rgbd_image.cu
        src/marching_cubes/mesh.cu
        src/marching_cubes/mesh_extractor.cu
        src/tsdfvh/heap.cu
        src/tsdfvh/hash_table.cu
        src/tsdfvh/tsdf_volume.cu
        src/tracker/tracker.cu
        OPTIONS -O3 -std=c++11 -Xcompiler -fPIC)

target_link_libraries(refusion eigen_wrapper ${CUDA_LIBRARIES})

add_executable(refusion_example src/example/example.cpp src/example/fr_parser.cpp)

target_link_libraries(refusion_example
        refusion
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        )

# Generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile @ONLY)
  add_custom_target(doc ALL
          ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          COMMENT "Generating API documentation with Doxygen" VERBATIM
          )
endif(DOXYGEN_FOUND)



cuda_add_library(${appname}-original SHARED refusion_library.cpp
        OPTIONS -O3 -std=c++11 -Xcompiler -fPIC)


target_link_libraries(${appname}-original
        "-Wl,--no-as-needed;${CUDA_LIBRARIES};-Wl,--as-needed"
        refusion
        ${OpenCV_LIBS}
        )

generate_slam_library(${PROJECT_NAME}-original "")