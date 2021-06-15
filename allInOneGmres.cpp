#include<math.h>
#include"KokkosKernels_IOUtils.hpp"
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlas.hpp>
#include<KokkosBlas3_trsm.hpp>
#include<KokkosSparse_spmv.hpp>

int main(int argc, char *argv[]) {

  typedef double                             ST;
  typedef int                               OT;
  typedef Kokkos::DefaultExecutionSpace     EXSP;

  //TODO: Should these really be layout left?
  using ViewVectorType = Kokkos::View<ST*,Kokkos::LayoutLeft, EXSP>;
  using ViewHostVectorType = Kokkos::View<ST*,Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using ViewMatrixType = Kokkos::View<ST**,Kokkos::LayoutLeft, EXSP>; 

  std::string filename("Laplace3D10.mtx"); // example matrix
  bool converged = false;
  int m = 50; //Max subspace size before restarting.
  double convTol = 1e-10; //Relative residual convergence tolerance.
  int cycLim = 50; //Max number of restarts or 'cycles'.
  int cycle = 0;
  int numIters;  //Number of iterations within the cycle before convergence.
  double trueRes; //Keep this in double regardless so we know how small error gets.
  double nrmB; 
  double relRes, shortRelRes;
  
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
  ViewVectorType Xiter("Xiter",n); //Intermediate solution at iterations before restart. 
  ViewVectorType B(Kokkos::ViewAllocateWithoutInitializing("B"),n);//right-hand side vec
  ViewVectorType Res(Kokkos::ViewAllocateWithoutInitializing("Res"),n); //Residual vector
  ViewVectorType Wj(Kokkos::ViewAllocateWithoutInitializing("W_j"),n); //Tmp work vector 1
  ViewVectorType TmpVec(Kokkos::ViewAllocateWithoutInitializing("TmpVec"),n); //Tmp work vector 2 //TODO is this needed?
  ViewHostVectorType GVec_h("GVec",m+1);
  ViewMatrixType GLsSoln("GLsSoln",m,1);//LS solution vec for Givens Rotation. Must be 2-D for trsm. 
  ViewMatrixType::HostMirror GLsSoln_h = Kokkos::create_mirror_view(GLsSoln); //This one is needed for triangular solve. 
  ViewHostVectorType CosVal_h("CosVal",m);
  ViewHostVectorType SinVal_h("SinVal",m);
  ViewMatrixType V(Kokkos::ViewAllocateWithoutInitializing("V"),n,m+1);
  ViewMatrixType VSub; //Subview of 1st m cols for updating soln. 

  ViewMatrixType Q("Q",m+1,m); //Q matrix for QR factorization of H //Only used in Arn Rec debug. 
  ViewMatrixType::HostMirror H_h = Kokkos::create_mirror_view(Q); //Make H into a host view of Q. 
  ViewMatrixType RFactor("RFactor",m,m);// Triangular matrix for QR factorization of H


  // Make rhs random.
  /*int rand_seed = std::rand();
  Kokkos::Random_XorShift64_Pool<> pool(rand_seed); //initially used seed 12371
  Kokkos::fill_random(B, pool, -1,1);*/

  // Make rhs ones so that results are repeatable:
  Kokkos::deep_copy(B,1.0);

  //Compute initial residuals:
  nrmB = KokkosBlas::nrm2(B);
  Kokkos::deep_copy(Res,B);
  KokkosSparse::spmv("N", 1.0, A, X, 0.0, Wj); // wj = Ax
  KokkosBlas::axpy(-1.0, Wj, Res); // res = res-Wj = b-Ax. 
  trueRes = KokkosBlas::nrm2(Res);
  relRes = trueRes/nrmB;
  std::cout << "Initial trueRes is : " << trueRes << std::endl;
    
  while( !converged && cycle < cycLim){
    GVec_h(0) = trueRes;

    // Run Arnoldi iteration:
    auto Vj = Kokkos::subview(V,Kokkos::ALL,0); //TODO:pre-declare this so no auto?
    Kokkos::deep_copy(Vj,Res);
    KokkosBlas::scal(Vj,1.0/trueRes,Vj); //V0 = V0/norm(V0)

    for (int j = 0; j < m; j++){
      KokkosSparse::spmv("N", 1.0, A, Vj, 0.0, Wj); //wj = A*Vj
      // Think this is MGS ortho, but 1 vector at a time?
      for (int i = 0; i <= j; i++){
        auto Vi = Kokkos::subview(V,Kokkos::ALL,i); 
        H_h(i,j) = KokkosBlas::dot(Vi,Wj);  //Vi^* Wj  //TODO is this the right order for cmplx dot product?
        KokkosBlas::axpy(-H_h(i,j),Vi,Wj);//wj = wj-Hij*Vi //Host
      }
      //Re-orthog:
/*      for (int i = 0; i <= j; i++){
        auto Vi = Kokkos::subview(V,Kokkos::ALL,i); 
        tmpScalar = KokkosBlas::dot(Vi,Wj); //Vi^* Wj
        KokkosBlas::axpy(-tmpScalar,Vi,Wj);//wj = wj-tmpScalar*Vi
        H_h(i,j) = H_h(i,j) + tmpScalar; 
        KokkosBlas::scal(TmpVec,H_h(i,j),Vi);//TmpVec = H_h(i,j)*Vi 
      }*/
      
      H_h(j+1,j) = KokkosBlas::nrm2(Wj); 
      if(H_h(j+1,j) < 1e-14){ //Host
        throw std::runtime_error("Lucky breakdown"); //TODO deal with this correctly? Did we check for convergence?
      }

      Vj = Kokkos::subview(V,Kokkos::ALL,j+1); 
      KokkosBlas::scal(Vj,1.0/H_h(j+1,j),Wj); // Wj = Vj/H(j+1,j)

      //Apply Givens rotation and compute shortcut residual:
      for(int i=0; i<j; i++){
        ST tempVal = CosVal_h(i)*H_h(i,j) + SinVal_h(i)*H_h(i+1,j);
        H_h(i+1,j) = -SinVal_h(i)*H_h(i,j) + CosVal_h(i)*H_h(i+1,j);
        H_h(i,j) = tempVal;
      }
      ST h1 = H_h(j,j);
      ST h2 = H_h(j+1,j);
      ST mod = (sqrt(h1*h1 + h2*h2));
      CosVal_h(j) = h1/mod;
      SinVal_h(j) = h2/mod;

      //Have to apply this Givens rotation outside the loop- requires the values adjusted in loop to compute cos and sin
      H_h(j,j) = CosVal_h(j)*H_h(j,j) + SinVal_h(j)*H_h(j+1,j);
      H_h(j+1,j) = 0.0; //Do this outside of loop so we get an exact zero here. 

      GVec_h(j+1) = GVec_h(j)*(-SinVal_h(j));
      GVec_h(j) = GVec_h(j)*CosVal_h(j);
      shortRelRes = abs(GVec_h(j+1))/nrmB;

      std::cout << "Shortcut relative residual for iteration " << j+(cycle*m) << " is: " << shortRelRes << std::endl;

      //If short residual converged, or time to restart, check true residual
      if( shortRelRes < convTol || j == m-1 ) {
        //Compute least squares soln with Givens rotation:
        auto GLsSolnSub_h = Kokkos::subview(GLsSoln_h,Kokkos::ALL,0); //Original view has rank 2, need a rank 1 here. 
        auto GVecSub_h = Kokkos::subview(GVec_h, Kokkos::make_pair(0,m));
        Kokkos::deep_copy(GLsSolnSub_h, GVecSub_h); //Copy LS rhs vec for triangle solve.
        auto GLsSolnSub2_h = Kokkos::subview(GLsSoln_h,Kokkos::make_pair(0,j+1),Kokkos::ALL);
        auto H_Sub_h = Kokkos::subview(H_h, Kokkos::make_pair(0,j+1), Kokkos::make_pair(0,j+1)); //TODO could change type from auto? 
        KokkosBlas::trsm("L", "U", "N", "N", 1.0, H_Sub_h, GLsSolnSub2_h); //GLsSoln = H\GLsSoln
        Kokkos::deep_copy(GLsSoln, GLsSoln_h);

        //Update solution and compute residual with Givens:
        VSub = Kokkos::subview(V,Kokkos::ALL,Kokkos::make_pair(0,j+1)); 
        Kokkos::deep_copy(Xiter,X); //Can't overwrite X with intermediate solution.
        auto GLsSolnSub3 = Kokkos::subview(GLsSoln,Kokkos::make_pair(0,j+1),0);
        KokkosBlas::gemv ("N", 1.0, VSub, GLsSolnSub3, 1.0, Xiter); //x_iter = x + V(1:j+1)*lsSoln
        KokkosSparse::spmv("N", 1.0, A, Xiter, 0.0, Wj); // wj = Ax
        Kokkos::deep_copy(Res,B); // Reset r=b.
        KokkosBlas::axpy(-1.0, Wj, Res); // r = b-Ax. 
        trueRes = KokkosBlas::nrm2(Res);
        relRes = trueRes/nrmB;
        std::cout << "True Givens relative residual for iteration " << j+(cycle*m) << " is : " << trueRes/nrmB << std::endl;
        numIters = j;

        if(relRes < convTol){
          converged = true;
          Kokkos::deep_copy(X, Xiter); //Final solution is the iteration solution.
          break; //End Arnoldi iteration. 
        }
      }

      // DEBUG: Print elts of H:
      /*std::cout << "Elements of H " <<std::endl;
        for (int i1 = 0; i1 < m+1; i1++){
        for (int j1 = 0; j1 < m; j1++){
        std::cout << H_h(i1,j1);
        }
        std::cout << std::endl;
        }*/

    }//end Arnoldi iter.

    /*//DEBUG: Check orthogonality of V:
    ViewMatrixType Vsm("Vsm", m+1, m+1);
      KokkosBlas::gemm("T","N", 1.0, V, V, 0.0, Vsm); // Vsm = V^T * V
      ViewVectorType nrmV("nrmV",m+1);
    KokkosBlas::nrm2(nrmV, Vsm); //nrmV = norm(Vsm)
    std::cout << "Norm of V^T V: " << std::endl;
    ViewVectorType::HostMirror nrmV_h = Kokkos::create_mirror_view(nrmV); 
    Kokkos::deep_copy(nrmV_h, nrmV);
    for (int i1 = 0; i1 < m+1; i1++){ std::cout << nrmV_h(i1) << " " ; } */

    /*//DEBUG: Check Arn Rec AV=VH
    Kokkos::deep_copy(Q,H_h);
    ViewMatrixType AV("AV", n, m);
    ViewMatrixType VH("VH", n, m);
    KokkosSparse::spmv("N", 1.0, A, VSub, 0.0, AV); //AV = A*V_m
    KokkosBlas::gemm("N","N", 1.0, V, Q, 0.0, VH); //VH = V*Q
    KokkosBlas::axpy(-1.0, AV, VH); //VH = VH-AV
    ViewVectorType nrmARec("ARNrm", m);
    ViewVectorType::HostMirror nrmARec_h = Kokkos::create_mirror_view(nrmARec); 
    KokkosBlas::nrm2( nrmARec, VH); //nrmARec = norm(VH)
    Kokkos::deep_copy(nrmARec_h, nrmARec);
    std::cout << "ArnRec norm check: " << std::endl;
    for (int i1 = 0; i1 < m; i1++){ std::cout << nrmARec_h(i1) << " " ; }
    std::cout << std::endl; */
    
    //Zero out Givens rotation vector and H matrix. 
    Kokkos::deep_copy(GVec_h,0);
    Kokkos::deep_copy(H_h,0); //TODO is this step really needed?  

    cycle++;

    //This is the end, or it's time to restart. Update solution to most recent vector.
    Kokkos::deep_copy(X, Xiter);
  }

  std::cout << "Ending true residual is: " << trueRes << std::endl;
  std::cout << "Ending relative residual is: " << relRes << std::endl;
  if( converged ){
    std::cout << "Solver converged! " << std::endl;
  }
  else{
    std::cout << "Solver did not converge. :( " << std::endl;
  }
  std::cout << "The solver completed " << (cycle-1)*m + numIters << " iterations." << std::endl;

  }
  Kokkos::finalize();

}

