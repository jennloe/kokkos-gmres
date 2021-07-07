cmake_minimum_required(VERSION 3.0.0)

project(Kokkos_Gmres)
find_package(KokkosKernels REQUIRED)

#add_library(gmres_lib hello.cpp)
#add_executable(hello.exe hello.cpp)
add_executable(ex_real_A.exe ex_real_A.cpp)
add_executable(ex_gemm.exe ex_performance_gemm.cpp)
add_executable(test_real_A.exe test_real_A.cpp)
add_executable(ex_cmplx_A.exe ex_cmplx_A.cpp)
add_executable(test_cmplx_A.exe test_cmplx_A.cpp)
# declare dependency on KokkosKernels
target_link_libraries(ex_real_A.exe PRIVATE Kokkos::kokkoskernels)
target_link_libraries(test_real_A.exe PRIVATE Kokkos::kokkoskernels)
target_link_libraries(ex_cmplx_A.exe PRIVATE Kokkos::kokkoskernels)
target_link_libraries(ex_gemm.exe PRIVATE Kokkos::kokkoskernels)
target_link_libraries(test_cmplx_A.exe PRIVATE Kokkos::kokkoskernels)

