cmake_minimum_required(VERSION 3.0.0)

project(Kokkos_Gmres)
find_package(KokkosKernels REQUIRED)

#add_library(gmres_lib hello.cpp)
add_executable(hello.exe hello.cpp)
add_executable(gmres.exe gmres.cpp)
# declare dependency on KokkosKernels
target_link_libraries(gmres.exe PRIVATE Kokkos::kokkoskernels)

#add_executable(calculator main.cpp)

#add_executable(duplicate_calculator main.cpp)

#add_executable(calculator main.cpp)



#target_link_libraries(calculator my_math my_print)
#target_link_libraries(duplicate_calculator my_math my_print)
#target_link_libraries(calculator my_math my_print)
