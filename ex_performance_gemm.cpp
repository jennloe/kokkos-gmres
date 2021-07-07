#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlas.hpp>


int main(int argc, char *argv[]) {

  typedef double ST1;
  typedef float ST2;
  typedef Kokkos::DefaultExecutionSpace     EXSP;
  typedef Kokkos::View<ST1**,Kokkos::LayoutLeft, EXSP> ViewDoubleType; 
  typedef Kokkos::View<ST2**,Kokkos::LayoutLeft, EXSP> ViewFloatType; 

  Kokkos::initialize();
  {

  int n = 5000;
  int iters = 1000;
  for (int i=1;i<argc;++i) {
    const std::string& token = argv[i];
    if (token == std::string("--size")) n = std::atoi(argv[++i]);
    if (token == std::string("--iters")) iters = std::atoi(argv[++i]);
    if (token == std::string("--help") || token == std::string("-h")){
      std::cout << "Kokkos GMRES solver options:" << std::endl
        << "--size   :  The size 'n' of the long vector. (Default: 5000)." << std::endl
        << "--iters   :  The number of times to repeat call to gemm. (Default: 1000)." << std::endl
        << "--help  -h    :  Display this help message." << std::endl 
        << "Example Call  :  ./ex_gemm.exe --size 300" << std::endl << std::endl;
      return 0; }
  }

  ViewDoubleType A(Kokkos::ViewAllocateWithoutInitializing("A"),n,50);
  ViewDoubleType B(Kokkos::ViewAllocateWithoutInitializing("B"),50,50);
  ViewDoubleType C(Kokkos::ViewAllocateWithoutInitializing("C"),n,50);

  ViewFloatType A2(Kokkos::ViewAllocateWithoutInitializing("A2"),n,50);
  ViewFloatType B2(Kokkos::ViewAllocateWithoutInitializing("B2"),50,50);
  ViewFloatType C2(Kokkos::ViewAllocateWithoutInitializing("C2"),n,50);

  int seed1 = 123;
  Kokkos::Random_XorShift64_Pool<> pool(seed1); 
  Kokkos::fill_random(A, pool, -1,1);
  Kokkos::fill_random(B, pool, -1,1);
  Kokkos::fill_random(A2, pool, -1,1);
  Kokkos::fill_random(B2, pool, -1,1);
  int seed2 = 456;
  Kokkos::Random_XorShift64_Pool<> pool2(seed2); 
  Kokkos::fill_random(C, pool2, -1,1);
  Kokkos::fill_random(C2, pool2, -1,1);

  //Try an initial loop:
  Kokkos::Tools::Experimental::pause_tools();
  for(int i=0; i < 222; i++){
    KokkosBlas::gemm("N","N", 1.0, A, B, 1.0, C); 
    KokkosBlas::gemm("N","N", 1.0, A2, B2, 1.0, C2); 
  }
  Kokkos::Tools::Experimental::resume_tools();

  for(int i=0; i < iters; i++){
    KokkosBlas::gemm("N","N", 1.0, A, B, 1.0, C); 
    KokkosBlas::gemm("N","N", 1.0, A2, B2, 1.0, C2); 
  }

  }
  Kokkos::finalize();
}
