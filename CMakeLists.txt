cmake_minimum_required(VERSION 3.0.0)

project(Kokkos_Gmres)
find_package(KokkosKernels REQUIRED)

#add_library(gmres_lib hello.cpp)
#add_executable(hello.exe hello.cpp)
add_executable(main.exe main.cpp)
add_executable(main_cmplx.exe main_cmplx.cpp)
add_executable(allInOneGmres.exe allInOneGmres.cpp)
# declare dependency on KokkosKernels
target_link_libraries(main.exe PRIVATE Kokkos::kokkoskernels)
target_link_libraries(main_cmplx.exe PRIVATE Kokkos::kokkoskernels)
target_link_libraries(allInOneGmres.exe PRIVATE Kokkos::kokkoskernels)

