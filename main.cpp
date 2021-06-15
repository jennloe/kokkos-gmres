#include<math.h>
#include"KokkosKernels_IOUtils.hpp"
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlas.hpp>
#include<KokkosBlas3_trsm.hpp>
#include<KokkosSparse_spmv.hpp>

#include"gmres.hpp"

int main(int argc, char *argv[]) {

  typedef double                             ST;
  typedef int                               OT;
  typedef Kokkos::DefaultExecutionSpace     EXSP;

  //TODO: Should these really be layout left?
  using ViewVectorType = Kokkos::View<ST*,Kokkos::LayoutLeft, EXSP>;
  using ViewHostVectorType = Kokkos::View<ST*,Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using ViewMatrixType = Kokkos::View<ST**,Kokkos::LayoutLeft, EXSP>; 

  std::string filename("Laplace3D10.mtx"); // example matrix
  int m = 50; //Max subspace size before restarting.
  double convTol = 1e-10; //Relative residual convergence tolerance.
  int cycLim = 50;

  for (int i=1;i<argc;++i) {
    const std::string& token = argv[i];
    if (token == std::string("--filename")) filename = argv[++i];
    if (token == std::string("--max-subsp")) m = std::atoi(argv[++i]);
    if (token == std::string("--max-restarts")) cycLim = std::atoi(argv[++i]);
    if (token == std::string("--tol")) convTol = std::stod(argv[++i]);
    if (token == std::string("--help") || token == std::string("-h")){
      std::cout << "Kokkos GMRES solver options:" << std::endl
        << "--filename    :  The name of a matrix market (.mtx) file for matrix A (Default Laplace3D10.mtx)." << std::endl
        << "--max-subsp   :  The maximum size of the Kyrlov subspace before restarting (Default 50)." << std::endl
        << "--max-restarts:  Maximum number of GMRES restarts (Default 50)." << std::endl
        << "--tol         :  Convergence tolerance.  (Default 1e-8)." << std::endl
        << "--help  -h    :  Display this help message." << std::endl 
        << "Example Call  :  ./Gmres.exe --filename Laplace3D100.mtx --tol 1e-5 --max-subsp 100 " << std::endl << std::endl;
      return 0; }
  }
  std::cout << "File to process is: " << filename << std::endl;
  std::cout << "Convergence tolerance is: " << convTol << std::endl;

  //Initialize Kokkos AFTER parsing parameters:
  Kokkos::initialize();
  {

  // Read in a matrix Market file and use it to test the Kokkos Operator.
  KokkosSparse::CrsMatrix<ST, OT, EXSP> A = 
    KokkosKernels::Impl::read_kokkos_crst_matrix<KokkosSparse::CrsMatrix<ST, OT, EXSP>>(filename.c_str()); 

  int n = A.numRows();
  ViewVectorType X("X",n); //Solution and initial guess
  ViewVectorType B(Kokkos::ViewAllocateWithoutInitializing("B"),n);//right-hand side vec

  // Make rhs random.
  /*int rand_seed = std::rand();
  Kokkos::Random_XorShift64_Pool<> pool(rand_seed); //initially used seed 12371
  Kokkos::fill_random(B, pool, -1,1);*/

  // Make rhs ones so that results are repeatable:
  Kokkos::deep_copy(B,1.0);

  gmres<ST, Kokkos::LayoutLeft, EXSP>(A, B, X, convTol, m, cycLim);

  //KokkosBlas::axpy(-1.0, , Res); // r = b-Ax. 
  //double endRes = KokkosBlas::nrm2(Res);
  //std::cout << "Verify from main: Ending residual is " << 

  }
  Kokkos::finalize();

}

