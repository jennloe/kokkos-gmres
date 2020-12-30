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
  ViewVectorType x("x",n); //Should init to zeros, right?
  ViewVectorType b(Kokkos::ViewAllocateWithoutInitializing("b"),n);
  ViewVectorType r(Kokkos::ViewAllocateWithoutInitializing("r"),n);
  ViewVectorType wj(Kokkos::ViewAllocateWithoutInitializing("w_j"),n);
  ViewVectorType tmpVec(Kokkos::ViewAllocateWithoutInitializing("tmpVec"),n);
  ViewVectorType lsVec("lsVec",m+1);
  ViewVectorType::HostMirror lsVec_h = Kokkos::create_mirror_view(lsVec);
  ViewVectorType::HostMirror GVec = Kokkos::create_mirror_view(lsVec); //Copy of this for Givens Rotation intermediate solve. 
  ViewVectorType::HostMirror tmpGVec = Kokkos::create_mirror_view(lsVec); //Copy of this for Givens Rotation intermediate solve. 
  ViewMatrixType lsTmp("lsTmp",m,1);
  ViewMatrixType::HostMirror lsTmp_h = Kokkos::create_mirror_view(lsTmp);
  ViewHostVectorType cosVal("cosVal",m);
  ViewHostVectorType sinVal("sinVal",m);

  //ViewMatrixType H("H",m+1,m);
  ViewMatrixType Q("Q",m+1,m);
  ViewMatrixType::HostMirror H = Kokkos::create_mirror_view(Q); //Make H into a host view of Q. 
  ViewMatrixType::HostMirror H_copy = Kokkos::create_mirror_view(Q); // Copy of H to transform with Givens Rotations.
  ViewMatrixType::HostMirror tmpH = Kokkos::create_mirror_view(Q); // Copy of H to transform with Givens Rotations.
  ViewMatrixType G_device("G_d", m+1, m+1); //For Givens rotation object. 
  ViewMatrixType::HostMirror G = Kokkos::create_mirror_view(G_device); // Not sure yet if we need this on host or device... so make both. 
  ViewMatrixType R("R",m,m);
  ViewMatrixType V(Kokkos::ViewAllocateWithoutInitializing("V"),n,m+1);
  ViewMatrixType VSub = Kokkos::subview(V,Kokkos::ALL,Kokkos::make_pair(0,m)); //Subview of 1st m cols for updating soln.

  double trueRes; //Keep this in double regardless so we know how small error gets.
  double nrmB; 
  double relRes;
  double val1, val2;


  // Make rhs random.
  /*int rand_seed = std::rand();
  Kokkos::Random_XorShift64_Pool<> pool(rand_seed); //initially used seed 12371
  Kokkos::fill_random(b, pool, -1,1);*/

  // Make rhs ones to replicate:
  Kokkos::deep_copy(b,1.0);
  nrmB = KokkosBlas::nrm2(b);
  Kokkos::deep_copy(r,b);

  KokkosSparse::spmv("N", 1.0, A, x, 0.0, wj); // wj = Ax
  KokkosBlas::axpy(-1.0, wj, r); // r = b-Ax. //TODO do we really need to store r separately?
  trueRes = KokkosBlas::nrm2(r);
  relRes = trueRes/nrmB;
  std::cout << "Initial trueRes is : " << trueRes << std::endl;
  val2 = trueRes; //Need this to check GVec
    
  while( relRes > convTol && cycle < cycLim){
    lsVec_h(0) = trueRes;
    GVec(0) = trueRes;

    //DEBUG: Print lsVec (rhs of ls prob)
    //std::cout << "lsVec elements: " << std::endl;
    //for (int i1 = 0; i1 < m+1; i1++){ std::cout << lsVec_h(i1) << " " ; }

    Kokkos::deep_copy(lsVec, lsVec_h);
    auto V0 = Kokkos::subview(V,Kokkos::ALL,0);
    Kokkos::deep_copy(V0,r);
    KokkosBlas::scal(V0,1.0/trueRes,V0); //V0 = V0/norm(V0)

    //Might need to move v0 normalize to here??
    //
    // Run Arnoldi iteration:

    // DEBUG: Print elts of H:
    /*for (int i1 = 0; i1 < m+1; i1++){
      for (int j1 = 0; j1 < m; j1++){
        std::cout << H(i1,j1);
      }
      std::cout << std::endl;
    }*/
    for (int j = 0; j < m; j++){
      auto Vj = Kokkos::subview(V,Kokkos::ALL,j); //TODO Could skip this one and use the v0 earlier and vj at end??
      KokkosSparse::spmv("N", 1.0, A, Vj, 0.0, wj); //wj = A*Vj
      // Think this is MGS ortho, but 1 vector at a time?
      for (int i = 0; i <= j; i++){
        auto Vi = Kokkos::subview(V,Kokkos::ALL,i); 
        H(i,j) = KokkosBlas::dot(Vi,wj);  //Host or device //TODO is this the right order for cmplx dot product?
        H_copy(i,j) = H(i,j);
        KokkosBlas::axpy(-H(i,j),Vi,wj);//wj = wj-Hij*Vi //Host
      }
      //Re-orthog:
/*      for (int i = 0; i <= j; i++){
        auto Vi = Kokkos::subview(V,Kokkos::ALL,i); 
        tmpScalar = KokkosBlas::dot(Vi,wj);
        KokkosBlas::axpy(-tmpScalar,Vi,wj);//wj = wj-tmpScalar*Vi
        H(i,j) = H(i,j) + tmpScalar; //Host
        H_copy(i,j) = H(i,j);
        KokkosBlas::scal(tmpVec,H(i,j),Vi);//tmpVec = H(i,j)*Vi //Host
      }*/
      
      //auto Hlast = Kokkos::subview(H,j+1,j);//TODO is this the right subview?? How does that indexing work?
      H(j+1,j) = KokkosBlas::nrm2(wj); //Host or device
      H_copy(j+1,j) = H(j+1,j);
      //std::cout << "Hlast is " << H(j+1,j) << std::endl;
      //bool myBool = H(j+1,j)<1e-14;
      //std::cout << "test bool is: " << myBool << std::endl;
      if(H(j+1,j) < 1e-14){ //Host
        //std::cout << "In the breakdonw if statement! " << std::endl;
        throw std::runtime_error("Lucky breakdown");
      }

      //Apply Givens rotation and compute short residual:
      for(int i=0; i<j; i++){
        ST tempVal = cosVal(i)*H_copy(i,j) + sinVal(i)*H_copy(i+1,j);
        H_copy(i+1,j) = -sinVal(i)*H_copy(i,j) + cosVal(i)*H_copy(i+1,j);
        H_copy(i,j) = tempVal;
      }
      auto H_copySub = Kokkos::subview(H_copy,Kokkos::make_pair(0,j+2),Kokkos::make_pair(0,j+1)); //Subview of part of H created so far.
      auto tmpHSub = Kokkos::subview(tmpH,Kokkos::make_pair(0,j+2),Kokkos::make_pair(0,j+1)); //Subview of part of H created so far.
      ST h1 = H_copy(j,j);
      ST h2 = H_copy(j+1,j);
      ST mod = (sqrt(h1*h1 + h2*h2));
      cosVal(j) = h1/mod;
      sinVal(j) = h2/mod;
      //std::cout << std::endl << "h1 and h2 are: " << h1 << "  " << h2 << " mod is: " << mod << std::endl;
      
      if( j > 0 ){
        G(j-1, j-1) = 1;
        G(j-1, j) = 0;
        G(j, j-1) = 0;
      }
      G(j,j) = cosVal(j);
      G(j, j+1) = sinVal(j);
      G(j+1, j) = -sinVal(j);
      G(j+1, j+1) = cosVal(j);
      auto GSub = Kokkos::subview(G,Kokkos::make_pair(0,j+2),Kokkos::make_pair(0,j+2)); //Subview of Givens rotator.

      //DEBUG: Print GVec
      int lenG = GVec.extent(0);
      /*std::cout << std::endl << "GVec before GEMV: " << std::endl;
      for (int i = 0; i < lenG; i++){
        std::cout << GVec(i) << std::endl;
      }
      std::cout << std::endl;*/

      //KokkosBlas::gemm("N","N",1.0,GSub,H_copySub,0.0,tmpHSub); //TODO This is calling Host GEMM.  Faster on GPU or no?
      //Kokkos::deep_copy(H_copySub,tmpHSub);//TODO could possibly remove this deep copy by alternating in/out

      //Have to apply this Givens rotation outside the loop- requires the values adjusted in loop to compute cos and sin
      H_copy(j,j) = cosVal(j)*H_copy(j,j) + sinVal(j)*H_copy(j+1,j);
      H_copy(j+1,j) = 0.0; //Do this outside of loop so we get an exact zero here. 

      //auto lsVecSub_h = Kokkos::subview(lsVec_h, Kokkos::make_pair(0,j+2));
      auto GVecSub = Kokkos::subview(GVec, Kokkos::make_pair(0,j+2));
      auto tmpGVecSub = Kokkos::subview(tmpGVec, Kokkos::make_pair(0,j+2));
      KokkosBlas::gemv("N", 1.0, GSub, GVecSub, 0.0, tmpGVecSub);
      Kokkos::deep_copy(GVecSub, tmpGVecSub); //TODO could possibly remove this deep copy by alternating in/out
      
      //DEBUG: Check first givens rot on GVec:
      //val1 = cosVal*val2;
      //val2 = -sinVal*val2;
      //std::cout << "cos*beta and -sin*beta are: " << cosVal*trueRes << -sinVal*trueRes << std::endl;
      //std::cout << "cos*beta and -sin*beta are: " << val1 << val2 << std::endl;
      //DEBUG: Print GVec
      /*std::cout << std::endl << "GVec after GEMV: " << std::endl;
      for (int i = 0; i < lenG; i++){
        std::cout << GVec(i) << std::endl;
      }*/
      std::cout << std::endl;

      std::cout << "Shortcut relative residual for iteration " << j+(cycle*50) << " is: " << abs(GVecSub(j+1))/nrmB << std::endl;

        

    // DEBUG: Print elts of GSub:
    /*int lenG0 = GSub.extent(0);
    int lenG1 = GSub.extent(1);
    for (int i1 = 0; i1 < lenG0; i1++){
      for (int j1 = 0; j1 < lenG1; j1++){
        std::cout << GSub(i1,j1) << "   " ;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;*/

    // DEBUG: Print elts of H_copySub:
    /*int len0 = H_copySub.extent(0);
    int len1 = H_copySub.extent(1);
    std::cout << std::endl;
    for (int i1 = 0; i1 < len0; i1++){
      for (int j1 = 0; j1 < len1; j1++){
        std::cout << H_copySub(i1,j1) << "   " ;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;*/



      Vj = Kokkos::subview(V,Kokkos::ALL,j+1); 
      KokkosBlas::scal(Vj,1.0/H(j+1,j),wj); //Host or maybe device?

    //Compute iteration least squares soln with QR:
    Kokkos::deep_copy(Q,H); //TODO Do we really need a copy, or can we reuse H? //copies to something on device....
    //Yes this ^^ is needed, now we made H a mirror view.  
    // Get subview of R so not dividing by zero.
   // auto RSub = Kokkos::subview(R,Kokkos::make_pair(0,j+1),Kokkos::make_pair(0,j+1)); 
    ViewMatrixType RSub("RSub", j+1,j+1);
    ViewMatrixType QSub = Kokkos::subview(Q,Kokkos::ALL,Kokkos::make_pair(0,j+1)); 
    //Compute QR factorization:
    mgsQR<EXSP, int, ViewMatrixType> (QSub,RSub);

    auto lsTmpSub = Kokkos::subview(lsTmp,Kokkos::ALL,0); //Original view has rank 2, need a rank 1 here. 
    KokkosBlas::gemv("T",1.0,Q,lsVec,0.0,lsTmpSub); 
    auto lsTmpSub2 = Kokkos::subview(lsTmp,Kokkos::make_pair(0,j+1),Kokkos::ALL);
    KokkosBlas::trsm("L", "U", "N", "N", 1.0, RSub, lsTmpSub2);

    //DEBUG: Check Short residual norm
    ViewMatrixType C3("C",m+1,m); //To test Q*R=H
    Kokkos::deep_copy(C3,H); //Need H on device to compare
    // DEBUG: Print elts of H:
    /*std::cout << "Elements of H at copy to C3:" <<std::endl;
    for (int i1 = 0; i1 < m+1; i1++){
      for (int j1 = 0; j1 < m; j1++){
        std::cout << H(i1,j1);
      }
      std::cout << std::endl;
    }*/
    ViewVectorType lsVecCpy("lsVecCpy",m+1);
    Kokkos::deep_copy(lsVecCpy,lsVec);
    // DEBUG: Print lsTmpSub
    /*std::cout << "Elts of lsTmpSub: " << std::endl;
    Kokkos::deep_copy(lsTmp_h, lsTmp);
    for (int i3 = 0; i3 < lsTmp_h.extent(0); i3++){
      std::cout << lsTmp_h(i3,0);
    }
    std::cout << std::endl;*/
    KokkosBlas::gemv("N",-1.0,C3,lsTmpSub,1.0,lsVecCpy); 
    ST shortRes = KokkosBlas::nrm2(lsVecCpy);
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
    Kokkos::deep_copy(Q,H); //TODO Do we really need a copy, or can we reuse H? //copies to something on device....
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
    mgsQR<EXSP, int, ViewMatrixType> (Q,R);

    auto lsTmpSub = Kokkos::subview(lsTmp,Kokkos::ALL,0); //Original view has rank 2, need a rank 1 here. 
    KokkosBlas::gemv("T",1.0,Q,lsVec,0.0,lsTmpSub); 
    KokkosBlas::trsm("L", "U", "N", "N", 1.0, R, lsTmp);

    //DEBUG: Check Short residual norm
    ViewMatrixType C3("C",m+1,m); //To test Q*R=H
    Kokkos::deep_copy(C3,H); //Need H on device to compare
    ViewVectorType lsVecCpy("lsVecCpy",m+1);
    Kokkos::deep_copy(lsVecCpy,lsVec);
    KokkosBlas::gemv("N",-1.0,C3,lsTmpSub,1.0,lsVecCpy); 
    ST shortRes = KokkosBlas::nrm2(lsVecCpy);
    std::cout << "Short residual is: " << shortRes << std::endl;

    //Update long solution and residual:
    KokkosBlas::gemv ("N", 1.0, VSub, lsTmpSub, 1.0, x); //x = x + V(1:m)*lsSoln

    //TODO Could avoid repeating this with a do-while loop?
    KokkosSparse::spmv("N", 1.0, A, x, 0.0, wj); // wj = Ax
    Kokkos::deep_copy(r,b); // Reset r=b.
    KokkosBlas::axpy(-1.0, wj, r); // r = b-Ax. //TODO do we really need to store r separately?
    trueRes = KokkosBlas::nrm2(r);
    relRes = trueRes/nrmB;
    std::cout << "Next trueRes is : " << trueRes << std::endl;
    std::cout << "Next relative residual is : " << relRes << std::endl;

    //Zero out Givens rotation vector. 
    Kokkos::deep_copy(GVec,0);

    //TODO Can probably remove this at the end.  Used so that we can run full LS problem to 
    //check short residual at each iteration.
    Kokkos::deep_copy(H,0);
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
