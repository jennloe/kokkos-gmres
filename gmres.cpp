#include<math.h>
#include"KokkosKernels_IOUtils.hpp"
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlas.hpp>
#include<KokkosBlas3_trsm.hpp>
#include<KokkosSparse_spmv.hpp>


template<typename exec_space, typename lno_t, typename Matrix>
void mgsQR(Matrix Q, Matrix R);

int main(int argc, char *argv[]) {

  typedef double                             ST;
  typedef int                               OT;
  typedef Kokkos::DefaultExecutionSpace     EXSP;

  Kokkos::initialize();
  {//TODO: Should these really be layout left?
  using ViewVectorType = Kokkos::View<ST*,Kokkos::LayoutLeft, EXSP>;
  using ViewHostVectorType = Kokkos::View<ST*,Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using ViewMatrixType = Kokkos::View<ST**,Kokkos::LayoutLeft, EXSP>; 

  std::string filename("Laplace3D10.mtx"); // example matrix
  //std::string filename("Identity50.mtx"); // example matrix
  bool converged = false;
  int m = 50; //Max subspace size.
  double convTol = 1e-10; //Keep in double.
  int cycLim = 100;
  int cycle = 0;
  
  //EXAMPLE: Parse cmnd line args: 
    /*for (int i=1;i<argc;++i) {
      const std::string& token = argv[i];
      if (token == std::string("-N")) N = std::atoi(argv[++i]);
      if (token == std::string("-B")) Blk = std::atoi(argv[++i]);
      if (token == std::string("-Task")) Task = std::atoi(argv[++i]);
      if (token == std::string("-TeamSize")) TeamSize = std::atoi(argv[++i]);
    }   
    printf(" :::: Testing (N = %d, Blk = %d, TeamSize = %d (0 is AUTO))\n", N, Blk, TeamSize); */ 
  
  for (int i=1;i<argc;++i) {
    const std::string& token = argv[i];
    if (token == std::string("--filename")) filename = argv[++i];
  }
  std::cout << "File to process is: " << filename << std::endl;


  // Read in a matrix Market file and use it to test the Kokkos Operator.
  KokkosSparse::CrsMatrix<ST, OT, EXSP> A = 
    KokkosKernels::Impl::read_kokkos_crst_matrix<KokkosSparse::CrsMatrix<ST, OT, EXSP>>(filename.c_str()); 

  int n = A.numRows();
  ViewVectorType X("X",n); //Solution and initial guess
  ViewVectorType B(Kokkos::ViewAllocateWithoutInitializing("B"),n);//right-hand side vec
  ViewVectorType Res(Kokkos::ViewAllocateWithoutInitializing("Res"),n); //Residual vector
  ViewVectorType Wj(Kokkos::ViewAllocateWithoutInitializing("W_j"),n); //Tmp work vector 1
  ViewVectorType TmpVec(Kokkos::ViewAllocateWithoutInitializing("TmpVec"),n); //Tmp work vector 2
  ViewVectorType LsVec("LsVec",m+1); //Small rhs of least-squares problem
  ViewVectorType::HostMirror LsVec_h = Kokkos::create_mirror_view(LsVec);
  ViewVectorType::HostMirror GVec_h = Kokkos::create_mirror_view(LsVec); //Copy of this for Givens Rotation intermediate solve. 
  ViewVectorType::HostMirror TmpGVec_h = Kokkos::create_mirror_view(LsVec); //Copy of this for Givens Rotation intermediate solve. 
  ViewMatrixType LsTmp("LsTmp",m,1);
  ViewMatrixType::HostMirror LsTmp_h = Kokkos::create_mirror_view(LsTmp);
  ViewHostVectorType CosVal_h("CosVal",m);
  ViewHostVectorType SinVal_h("SinVal",m);

  //ViewMatrixType H("H",m+1,m);
  ViewMatrixType Q("Q",m+1,m); //Q matrix for QR factorization of H
  ViewMatrixType::HostMirror H_h = Kokkos::create_mirror_view(Q); //Make H into a host view of Q. 
  ViewMatrixType::HostMirror H_copy_h = Kokkos::create_mirror_view(Q); // Copy of H to transform with Givens Rotations.
  ViewMatrixType RFactor("RFactor",m,m);// Triangular matrix for QR factorization of H
  ViewMatrixType V(Kokkos::ViewAllocateWithoutInitializing("V"),n,m+1);
  ViewMatrixType VSub = Kokkos::subview(V,Kokkos::ALL,Kokkos::make_pair(0,m)); //Subview of 1st m cols for updating soln.

  double trueRes; //Keep this in double regardless so we know how small error gets.
  double nrmB; 
  double relRes;


  // Make rhs random.
  /*int rand_seed = std::rand();
  Kokkos::Random_XorShift64_Pool<> pool(rand_seed); //initially used seed 12371
  Kokkos::fill_random(B, pool, -1,1);*/

  // Make rhs ones to replicate:
  Kokkos::deep_copy(B,1.0);
  nrmB = KokkosBlas::nrm2(B);
  Kokkos::deep_copy(Res,B);

  KokkosSparse::spmv("N", 1.0, A, X, 0.0, Wj); // wj = Ax
  KokkosBlas::axpy(-1.0, Wj, Res); // r = b-Ax. //TODO do we really need to store r separately?
  trueRes = KokkosBlas::nrm2(Res);
  relRes = trueRes/nrmB;
  std::cout << "Initial trueRes is : " << trueRes << std::endl;
    
  while( relRes > convTol && cycle < cycLim){
    LsVec_h(0) = trueRes;
    GVec_h(0) = trueRes;

    //DEBUG: Print lsVec (rhs of ls prob)
    //std::cout << "lsVec elements: " << std::endl;
    //for (int i1 = 0; i1 < m+1; i1++){ std::cout << LsVec_h(i1) << " " ; }

    Kokkos::deep_copy(LsVec, LsVec_h);
    auto V0 = Kokkos::subview(V,Kokkos::ALL,0);
    Kokkos::deep_copy(V0,Res);
    KokkosBlas::scal(V0,1.0/trueRes,V0); //V0 = V0/norm(V0)

    //Might need to move v0 normalize to here??
    //
    // Run Arnoldi iteration:

    // DEBUG: Print elts of H:
    /*for (int i1 = 0; i1 < m+1; i1++){
      for (int j1 = 0; j1 < m; j1++){
        std::cout << H_h(i1,j1);
      }
      std::cout << std::endl;
    }*/
    for (int j = 0; j < m; j++){
      auto Vj = Kokkos::subview(V,Kokkos::ALL,j); //TODO Could skip this one and use the v0 earlier and vj at end??
      KokkosSparse::spmv("N", 1.0, A, Vj, 0.0, Wj); //wj = A*Vj
      // Think this is MGS ortho, but 1 vector at a time?
      for (int i = 0; i <= j; i++){
        auto Vi = Kokkos::subview(V,Kokkos::ALL,i); 
        H_h(i,j) = KokkosBlas::dot(Vi,Wj);  //Host or device //TODO is this the right order for cmplx dot product?
        H_copy_h(i,j) = H_h(i,j);
        KokkosBlas::axpy(-H_h(i,j),Vi,Wj);//wj = wj-Hij*Vi //Host
      }
      //Re-orthog:
/*      for (int i = 0; i <= j; i++){
        auto Vi = Kokkos::subview(V,Kokkos::ALL,i); 
        tmpScalar = KokkosBlas::dot(Vi,Wj);
        KokkosBlas::axpy(-tmpScalar,Vi,Wj);//wj = wj-tmpScalar*Vi
        H_h(i,j) = H_h(i,j) + tmpScalar; //Host
        H_copy_h(i,j) = H_h(i,j);
        KokkosBlas::scal(TmpVec,H_h(i,j),Vi);//TmpVec = H_h(i,j)*Vi //Host
      }*/
      
      H_h(j+1,j) = KokkosBlas::nrm2(Wj); //Host or device
      H_copy_h(j+1,j) = H_h(j+1,j);
      if(H_h(j+1,j) < 1e-14){ //Host
        throw std::runtime_error("Lucky breakdown");
      }

      //Apply Givens rotation and compute short residual:
      for(int i=0; i<j; i++){
        ST tempVal = CosVal_h(i)*H_copy_h(i,j) + SinVal_h(i)*H_copy_h(i+1,j);
        H_copy_h(i+1,j) = -SinVal_h(i)*H_copy_h(i,j) + CosVal_h(i)*H_copy_h(i+1,j);
        H_copy_h(i,j) = tempVal;
      }
      auto H_copySub_h = Kokkos::subview(H_copy_h,Kokkos::make_pair(0,j+2),Kokkos::make_pair(0,j+1)); //Subview of part of H created so far.
      ST h1 = H_copy_h(j,j);
      ST h2 = H_copy_h(j+1,j);
      ST mod = (sqrt(h1*h1 + h2*h2));
      CosVal_h(j) = h1/mod;
      SinVal_h(j) = h2/mod;
      
      //DEBUG: Print GVec
      int lenG = GVec_h.extent(0);
      /*std::cout << std::endl << "GVec before GEMV: " << std::endl;
      for (int i = 0; i < lenG; i++){
        std::cout << GVec_h(i) << std::endl;
      }
      std::cout << std::endl;*/

      //Have to apply this Givens rotation outside the loop- requires the values adjusted in loop to compute cos and sin
      H_copy_h(j,j) = CosVal_h(j)*H_copy_h(j,j) + SinVal_h(j)*H_copy_h(j+1,j);
      H_copy_h(j+1,j) = 0.0; //Do this outside of loop so we get an exact zero here. 

      GVec_h(j+1) = GVec_h(j)*(-SinVal_h(j));
      GVec_h(j) = GVec_h(j)*CosVal_h(j);

      //DEBUG: Print GVec
      /*std::cout << std::endl << "GVec after GEMV: " << std::endl;
      for (int i = 0; i < lenG; i++){
        std::cout << GVec_h(i) << std::endl;
      }*/
      std::cout << std::endl;

      std::cout << "Shortcut relative residual for iteration " << j+(cycle*50) << " is: " << abs(GVec_h(j+1))/nrmB << std::endl;

    // DEBUG: Print elts of H_copySub:
    /*int len0 = H_copySub_h.extent(0);
    int len1 = H_copySub_h.extent(1);
    std::cout << std::endl;
    for (int i1 = 0; i1 < len0; i1++){
      for (int j1 = 0; j1 < len1; j1++){
        std::cout << H_copySub_h(i1,j1) << "   " ;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;*/

      Vj = Kokkos::subview(V,Kokkos::ALL,j+1); 
      KokkosBlas::scal(Vj,1.0/H_h(j+1,j),Wj); //Host or maybe device?

    //Compute iteration least squares soln with QR:
    Kokkos::deep_copy(Q,H_h); //TODO Do we really need a copy, or can we reuse H? //copies to something on device....
    //Yes this ^^ is needed, now we made H a mirror view.  
    ViewMatrixType RFactorSm("RFactorSm", j+1,j+1);
    ViewMatrixType QSub = Kokkos::subview(Q,Kokkos::ALL,Kokkos::make_pair(0,j+1)); 
    //Compute QR factorization:
    mgsQR<EXSP, int, ViewMatrixType> (QSub,RFactorSm);

    auto LsTmpSub = Kokkos::subview(LsTmp,Kokkos::ALL,0); //Original view has rank 2, need a rank 1 here. 
    KokkosBlas::gemv("T",1.0,Q,LsVec,0.0,LsTmpSub); 
    auto LsTmpSub2 = Kokkos::subview(LsTmp,Kokkos::make_pair(0,j+1),Kokkos::ALL);
    KokkosBlas::trsm("L", "U", "N", "N", 1.0, RFactorSm, LsTmpSub2);

    //DEBUG: Check Short residual norm
    ViewMatrixType C3("C",m+1,m); //To test Q*R=H
    Kokkos::deep_copy(C3,H_h); //Need H on device to compare
    // DEBUG: Print elts of H:
    /*std::cout << "Elements of H at copy to C3:" <<std::endl;
    for (int i1 = 0; i1 < m+1; i1++){
      for (int j1 = 0; j1 < m; j1++){
        std::cout << H_h(i1,j1);
      }
      std::cout << std::endl;
    }*/
    ViewVectorType LsVecCpy("LsVecCpy",m+1);
    Kokkos::deep_copy(LsVecCpy,LsVec);
    // DEBUG: Print lsTmpSub
    /*std::cout << "Elts of LsTmpSub: " << std::endl;
    Kokkos::deep_copy(LsTmp_h, LsTmp);
    for (int i3 = 0; i3 < LsTmp_h.extent(0); i3++){
      std::cout << LsTmp_h(i3,0);
    }
    std::cout << std::endl;*/
    KokkosBlas::gemv("N",-1.0,C3,LsTmpSub,1.0,LsVecCpy); 
    ST shortRes = KokkosBlas::nrm2(LsVecCpy);
    std::cout << "Short relative residual for iteration " << j+(cycle*50) << " is: " << shortRes/nrmB << std::endl;


    }//end Arnoldi iter.

    //DEBUG: Check orthogonality of V:
    /*ViewMatrixType Vsm("Vsm", m+1, m+1);
    KokkosBlas::gemm("T","N", 1.0, V, V, 0.0, Vsm);
    ViewVectorType nrmV("nrmV",m+1);
    KokkosBlas::nrm2(nrmV, Vsm);
    std::cout << "Norm of V^T V: " << std::endl;
    ViewVectorType::HostMirror nrmV_h = Kokkos::create_mirror_view(nrmV); 
    Kokkos::deep_copy(nrmV_h, nrmV);
    //for (int i1 = 0; i1 < m+1; i1++){ std::cout << nrmV_h(i1) << " " ; } */


    //Compute least squares soln:
    Kokkos::deep_copy(Q,H_h); //TODO Do we really need a copy, or can we reuse H? //copies to something on device....
    //Yes this ^^ is needed, now we made H a mirror view.  

    /*//DEBUG: Check Arn Rec AV=VH
    ViewMatrixType AV("AV", n, m);
    ViewMatrixType VH("VH", n, m);
    KokkosSparse::spmv("N", 1.0, A, VSub, 0.0, AV); 
    KokkosBlas::gemm("N","N", 1.0, V, Q, 0.0, VH);
    KokkosBlas::axpy(-1.0, AV, VH); //VH = VH-AV
    ViewVectorType nrmARec("ARNrm", m);
    ViewVectorType::HostMirror nrmARec_h = Kokkos::create_mirror_view(nrmARec); 
    KokkosBlas::nrm2( nrmARec, VH);
    Kokkos::deep_copy(nrmARec_h, nrmARec);
    std::cout << "ArnRec norm check: " << std::endl;
    for (int i1 = 0; i1 < m; i1++){ std::cout << nrmARec_h(i1) << " " ; }
    std::cout << std::endl; */

    //Compute QR factorization:
    mgsQR<EXSP, int, ViewMatrixType> (Q,RFactor);

    auto LsTmpSub = Kokkos::subview(LsTmp,Kokkos::ALL,0); //Original view has rank 2, need a rank 1 here. 
    KokkosBlas::gemv("T",1.0,Q,LsVec,0.0,LsTmpSub); 
    KokkosBlas::trsm("L", "U", "N", "N", 1.0, RFactor, LsTmp);

    //DEBUG: Check Short residual norm
    ViewMatrixType C3("C",m+1,m); //To test Q*R=H
    Kokkos::deep_copy(C3,H_h); //Need H on device to compare
    ViewVectorType LsVecCpy("LsVecCpy",m+1);
    Kokkos::deep_copy(LsVecCpy,LsVec);
    KokkosBlas::gemv("N",-1.0,C3,LsTmpSub,1.0,LsVecCpy); 
    ST shortRes = KokkosBlas::nrm2(LsVecCpy);
    std::cout << "Short residual is: " << shortRes << std::endl;

    //Update long solution and residual:
    KokkosBlas::gemv ("N", 1.0, VSub, LsTmpSub, 1.0, X); //x = x + V(1:m)*lsSoln

    //TODO Could avoid repeating this with a do-while loop?
    KokkosSparse::spmv("N", 1.0, A, X, 0.0, Wj); // wj = Ax
    Kokkos::deep_copy(Res,B); // Reset r=b.
    KokkosBlas::axpy(-1.0, Wj, Res); // r = b-Ax. //TODO do we really need to store r separately?
    trueRes = KokkosBlas::nrm2(Res);
    relRes = trueRes/nrmB;
    std::cout << "Next trueRes is : " << trueRes << std::endl;
    std::cout << "Next relative residual is : " << relRes << std::endl;

    //Zero out Givens rotation vector. 
    Kokkos::deep_copy(GVec_h,0);

    //TODO Can probably remove this at the end.  Used so that we can run full LS problem to 
    //check short residual at each iteration.
    Kokkos::deep_copy(H_h,0);
    cycle++;
  }

  std::cout << "Ending true residual is: " << trueRes << std::endl;
  std::cout << "Ending relative residual is: " << relRes << std::endl;
  if( relRes < convTol )
    std::cout << "Solver converged! " << std::endl;
  else
    std::cout << "Solver did not converge. :( " << std::endl;
  std::cout << "Number of cycles completed is " << cycle << std::endl;
  std::cout << "which corresponds to " << cycle*m << " iterations." << std::endl;

  }
  Kokkos::finalize();

}


template<typename exec_space, typename lno_t, typename Matrix>
void mgsQR(Matrix Q, Matrix R)
{
  lno_t k = Q.extent(1);
  //Set R = I(k)
  auto Rhost = Kokkos::create_mirror_view(R);
  for(lno_t i = 0; i < k; i++)
  {
    for(lno_t j = 0; j < k; j++)
      Rhost(i, j) = 0;
    Rhost(i, i) = 1;
  }
  Kokkos::deep_copy(R, Rhost);
  for(lno_t i = 0; i < k; i++)
  {
    auto QcolI = Kokkos::subview(Q, Kokkos::ALL(), i);
    //normalize column i
    double colNorm = KokkosBlas::nrm2(QcolI);
    KokkosBlas::scal(QcolI, 1.0 / colNorm, QcolI);
    //scale up R row i by inorm
    auto RrowI = Kokkos::subview(R, i, Kokkos::ALL());
    KokkosBlas::scal(RrowI, colNorm, RrowI);
    for(lno_t j = i + 1; j < k; j++)
    {
      auto QcolJ = Kokkos::subview(Q, Kokkos::ALL(), j);
      auto RrowJ = Kokkos::subview(R, j, Kokkos::ALL());
      //orthogonalize QcolJ against QcolI
      double d = KokkosBlas::dot(QcolI, QcolJ);
      KokkosBlas::axpby(-d, QcolI, 1, QcolJ);
      KokkosBlas::axpby(d, RrowJ, 1, RrowI);
    }
  }
}
