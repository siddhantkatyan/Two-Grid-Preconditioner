/* Deflated Two-Grid Solve as a preconditioner for GMRES */

#ifndef V3D_TG_GMRES_SOLVE_H
#define V3D_TG_GMRES_SOLVE_H

// C++ libraries
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstring>
#include <ctime>
#include <cmath>

using namespace std;    // C++ standard libary namespace
using namespace V3D;

// umfpack libraries
#include "cs.h"
#include "umfpack.h"
//#include "util.h"

//mkl libraries
#include "mkl.h"
#include "mkl_solvers_ee.h"

namespace V3D
{

// macros
//#define max(a, b) (a) < (b) ? (b): (a)  // for eigenvalue solve routine
#define sizelu 128                       // for ILU 
#define sizeG  8721 //hardcoding it for the time being
#define size_MKL_IPAR 128


/********* Two-grid Solve Function ***********/


void tg_solve(int num_cols,int kk, double *acsr,int *ia,int *ja, double *bilu0, int *bia, int *bja, double E[], double X[], double *x_init, double *x_2)
{

  /* 1st Step : Pre-smoothing  */

//  cout << "*******Starting Pre-smoothing******" << endl;
//  cout << endl;
//  cout << "*******Solving t = U\\(L\\x) ....*******"<< endl;
//  cout << endl;
  //printf("  Solving t = U\\(L\\x) system   \n");

  // Paramteres for triangular solver routine - mkl_cspblas_scsrtrsv

  double *t_0 = new double[num_cols];
  double *t_1 = new double[num_cols];

  char uplo = 'l';        // lower triangular
  char nonunit = 'n';     // diagonal is not unit triangular
  char transa = 'n';      // not tranpose
  MKL_INT ivar1;
  ivar1 = num_cols;       // ivar1 = sizeB = num_cols

  mkl_dcsrtrsv(&uplo,&transa,&nonunit,&ivar1,bilu0,bia,bja,x_init,t_0);
  /* Computation of L\X complete: L\x = t_0  */


  uplo = 'u';        // upper triangular
  nonunit = 'n';     // diagonal is not unit triangular
  transa = 'n';      // not tranpose

  /* Compute U\(t_0); t_0 = L\x */
  mkl_dcsrtrsv(&uplo,&transa,&nonunit,&ivar1,bilu0,bia,bja,t_0,t_1);
  /*  Solution obtained in t_1  */

//  printf("*******Pre-smoothing Complete!!!********\n\n");

//  cout << "Checking first few entries of Pre-smoothing solution" << endl;
//  for(int l = 0; l<5; l++)
//    cout << t_1[l] << endl;

//  printf("\n");

  /* 2nd Step: Coarse Grid Solve:  g = P* (Ac \ (P'*x))  */


//  cout << "*******Starting Coarse Grid-Solve*********\n"<<endl;

  //1st perform the innermost matrix vector multiplication i.e P'*x

  // Initialize the parameters for Matrix Vector Multiplication Routine : dgemv

  double *x_0 = new double[kk];  // size = number of eigenvalues

//MKL_INT ldy  = k;     /* Leading dimension for destination array in GEMM */
  MKL_INT ldx  = num_cols;     /* Leading dimension for source arrays in GEMM */
  char trans = 'T';     /* Transpose of the matrix */
  double one   = 1.0;   /* alpha parameter for GEMM */
  double zero  = 0.0;   /* alpha parameter for GEMM */
  MKL_INT incx = 1;     /* Specifies the increment for the elements of x */
  MKL_INT incy = 1;     /* Specifies the increment for the elements of y */

  

  // Call dgemv routine 
  dgemv(
        &trans,            /* IN: 'T', transposed case*/
        &num_cols,               /* IN: Number of rows in matrix P */
        &kk,               /* IN: Number of columns in matrix P = Number of Eigenvalues*/
        &one,             /* IN: alpha = 1.0 */
        X,                /* IN: Source matrix for GEMV, will be transposed */
        &ldx,             /* IN: Leading dimension of Matrix */
        x_init,          /* IN: Vector for GEMV */
        &incx,            /* IN: Specifies the increment for the elements of x*/
        &zero,            /* IN: beta = 0.0 */
        x_0,              /* OUT: Destination */
        &incy             /* IN: Specifies the increment for the elements of y*/
        );

//  cout << "*****Innermost P'*x Multiplication Complete !!! *****\n" <<endl;

//  cout << "Result of Matrix-Vector Multiplication is 5X1 vector" << endl;
  
//  for (int l =0; l < kk; l++)
//    cout << *(x_0+l) << endl;
//  printf("\n");


  // Performs pointwise element by element division of vector x_0 by vector Ac
  //vsDiv( n, a, b, y );
//   cout <<"\n*******Solving x_1 = Ac\\(P'*x) ....*******\n"<< endl;

  double *x_1 = new double[kk];
  vdDiv(kk, x_0, E, x_1);  // number of elements = k (number of eigenvalues) 

//  cout <<"******Solution x_1 = Ac\\x_0 is 5X1 vector ...******\n" <<endl;

//  for (int l =0; l < kk; l++)
//    cout << *(x_1+l) << endl;

//  printf("\n");

  /*   
    Now compute matrix -vector multiplication :g = P * x_1 ; x1 = (Ac \ (P'*x));
  */

  // Initialize the parameters for Matrix Vector Multiplication Routine : dgemv

 // double *x_2 = new double[N];  // size = number of eigenvalues

//MKL_INT ldy  = k;     /* Leading dimension for destination array in GEMM */
  MKL_INT ldx1  = num_cols;     /* Leading dimension for source arrays in GEMM */
  char trans1 = 'N';     /* Transpose of the matrix */
  double one1   = 1.0;   /* alpha parameter for GEMM */
  double zero1  = 0.0;   /* alpha parameter for GEMM */
  MKL_INT incx1 = 1;     /* Specifies the increment for the elements of x */
  MKL_INT incy1 = 1;     /* Specifies the increment for the elements of y */

  

  // Call dgemv routine 
  dgemv(
        &trans1,            /* IN: 'N',not transposed case*/
        &num_cols,               /* IN: Number of rows in matrix P */
        &kk,               /* IN: Number of columns in matrix P = Number of Eigenvalues*/
        &one1,             /* IN: alpha = 1.0 */
        X,                /* IN: Source matrix for GEMV, will NOT be transposed */
        &ldx1,             /* IN: Leading dimension of Matrix */
        x_1,              /* IN: Vector for GEMV */
        &incx1,            /* IN: Specifies the increment for the elements of x*/
        &zero1,            /* IN: beta = 0.0 */
        x_2,              /* OUT: Destination */
        &incy1             /* IN: Specifies the increment for the elements of y*/
        );

//  cout << "\nResult of Mat-Vec Multiplication: P*x_1 is num_cols X 1 first few entries are....\n" << endl;
  
//  for (int l =0; l < 10; l++)
//    cout << *(x_2+l) << endl;
//  printf("\n");

  // Compute : q(x_3) = A*(x_2); 
//  cout << "\n******Computing x_3 = A*x_2********\n" << endl;

  // paramters initialization for CSR matrix * vector multiplication
  char  matdescra1[4];
    
  char transa1  = 'n';           // not-transpose
  matdescra1[0] = 'S';    // Symmetric Matrix
  matdescra1[1] = 'L';
  matdescra1[2] = 'N';    // 
  matdescra1[3] = 'F';  
  double  alpha = 1.0, beta = 0.0;
  
  double *x_3  = new double[num_cols];

  // Call to Routine CSR Mat-Vec Multiplication 
  mkl_dcsrmv(&transa1, &num_cols, &num_cols, &alpha, matdescra1, acsr, ja, ia, ia+1, x_2, &beta, x_3);

  // Computation : q(x_3) = A*g; where g = (x_2) Complete;
//   cout << "\nResult of Mat-Vec Multiplication: A*x_2 is num_cols X 1 first few entries are....\n" << endl;
  
//  for (int l =0; l < 10; l++)
//    cout << *(x_3+l) << endl;
//  printf("\n");



  // *********Last Step : t + g - U\(L\q)***********
  // q_0 = L\q where q = x_3
  // q_1 = U\q_0

//  cout << "*******Solving q_0 = U\\(L\\x_3) ....*******\n"<< endl;

  // allocate memory for solution vectors
  double *q_0 = new double[num_cols];   
  double *q_1 = new double[num_cols];   // N = num_cols in JTJ matrix

  //Initializing Parameters for Triangular Solve Routine
  char uplo2 = 'l';        // lower triangular
  char nonunit2 = 'n';     // diagonal is not unit triangular
  char transa2 = 'n';      // not tranpose

  mkl_dcsrtrsv(&uplo2,&transa2,&nonunit2,&ivar1,bilu0,bia,bja,x_3,q_0);
  /* Computation of L\X complete: L\x_3 = q_0  */


  uplo2 = 'u';        // upper triangular
  nonunit2 = 'n';     // diagonal is not unit triangular
  transa2 = 'n';      // not tranpose

  /* Compute U\(q_0); t_0 = L\x */
  mkl_dcsrtrsv(&uplo2,&transa2,&nonunit2,&ivar1,bilu0,bia,bja,q_0,q_1);
  /*  Solution obtained in q_1  */

//  printf("*******Solution Complete!!!********\n\n");

//  cout << "Checking first few entries of q_1 = U\\(L\\x_3) solution ..." << endl;
//  for(int l = 0; l<5; l++)
//    cout << q_1[l] << endl;

//  printf("\n");


//  cout << "*******Addition of vector: t_1 + x_2********\n"<<endl;
  // Initializing parameters for sparse vector-vector addition 
  double alpha2  = 1.0;
  int incx2      = 1; 
  int incy2      = 1;

  // t + g; where g = x_2; t= t_1

  //Computes a vector-scalar prodct and adds the result to a vector
  daxpy(&num_cols, &alpha2, t_1, &incx2, x_2, &incy2);
  // addition of t+g complete, x_2 = x_2 + t_1
//  cout << "Checking first few entries of Addition ...." << endl;
//  for(int l = 0; l<5; l++)
//    cout << x_2[l] << endl;

//  printf("\n");


  /* Initializing parameters for sparse vector-vector subtraction 
       x_2 - q_1 
       where x_2 = t+g
       and q_1 = U\(L\q)

  */
//  cout << "*******Subtraction of vector: x_2 - q_1********\n"<<endl;
  double   alpha3 = -1;      // for subtraction
  int incx3 = 1, incy3 = 1;

  daxpy(&num_cols, &alpha3, q_1, &incx3, x_2, &incy3);      // x_2 - q_1 written in form -q_1 + x_2
    // Subtraction Complete: x_2 = x_2 - q_1

//  cout << "Checking first few entries of Subtraction ...." << endl;
//  for(int l = 0; l<5; l++)
//    cout << x_2[l] << endl;

//  printf("\n");

  /*
      Two - Grid Solve Complete 
      Solution obtained in x_2

      Now pass this preconditioned solution to GMRES

  */

  delete [] t_0; 
  delete [] t_1; 
  delete [] x_0; 
  delete [] x_1; 
  delete [] x_3;
  delete [] q_0;
  delete [] q_1;

  return;


}


/****************End of tg_solve Function***************/



void tg_gmres_solve(int num_cols,int ncc,int *colStarts,int *rowIdxs,double *values,double *Jt_e,double *delta)
{
  

//  double* JTe; //rhs
  int i;
//  int num_rows;
//  int num_cols; // no of columns
//  int ncc = 0; // no of non zeros
  int *null = ( int * ) NULL;
  double *solve_null = ( double * ) NULL;
  void *Numeric, *Numeric_D,*Numeric_MSC;
//  string prefix = "JTJ49_1";
  double r;
  int status,sym_status,num_status,solve_status;
  void* Symbolic,*Symbolic_D,*Symbolic_MSC;
  int sizeD;
  //int sizeG;
  int sizeB;   // Block Diagonal size
  int j,k;
  int nzD = 0;
  int nzG = 0;
  int nzL = 0; // nzL = nzU
  int nzB = 0; // Non zero in Block Diagonal
  int iterD = 0;
  int iterL = 0;
  int iterG = 0;
  int iterB = 0; 
  int count = 1;
  //creating structures of cs_diPARSE
  cs_di* A = new cs_di;
//  cs_di* MSC = new cs_di;
  cs_di* D = new cs_di;
  cs_di* L = new cs_di;
  cs_di* G = new cs_di;
  cs_di* U = new cs_di;
  cs_di* B = new cs_di;


  //string line;
//  string rhs_filename = "JTe49_1.txt";


  // getting the matrix size: n -> no of cols, ncc -> no of non zeros
//  cc_header_read ( prefix, ncc, num_cols );
//  num_cols = num_cols+1;                          //DOUBT!!!!!!!!
//  cout << "\nNo of non zeros = "<< ncc << "\n";
//  cout << "\nNo of columns = "<< num_cols << "\n";

  //size of the D matrix
  sizeD = num_cols - sizeG; 
  //size of the B matrix
  sizeB = num_cols;

  A->nzmax = ncc;
  A->m = num_cols;
  A->n = num_cols; // since square matrix
  A->nz = -1;

  //Allocate space for rhs
 // JTe = new double[num_cols];

  // reading the rhs
 // r8vec_data_read ( rhs_filename, num_cols, JTe);


  //Allocate space for the coefficient matrix
  A->x = new double[ncc];
  A->p = new int[num_cols+1];
  A->i = new int[ncc];

  //read the matrix data
  for(int q = 0; q < num_cols; q++) A->p[q] = colStarts[q];
  for(int q = 0; q < ncc; q++) A->i[q] = rowIdxs[q];
  for(int q = 0; q < ncc; q++) A->x[q] = values[q];
//  cc_data_read ( prefix, ncc, num_cols, A->i, A->p, A->x );
  A->p[num_cols] = ncc;

//  cout << "\nFile read complete!\n";
  /**************File Read in CSC format Complete***************/


  /**********Converting A from CSC to CSR*****************/
  MKL_INT ivar;
  char cvar;
  ivar = num_cols;
  cvar = 'N'; //no transpose
  MKL_INT job[6] = {1,1,0,0,0,1};    // various parameters for conversion
  double *acsr = new double[ncc];    // values
  MKL_INT *ja = new MKL_INT[ncc];    // column
  MKL_INT *ia = new MKL_INT[ivar+1]; // rows
  MKL_INT info1;                      // status

  // call to routine dcsrcsr
  mkl_dcsrcsc(job,&ivar,acsr,ja,ia,A->x,A->i,A->p,&info1);

//  cout << "\n Conversion info A : "<< info1 << "\n";

  /**************Matrix Conversion in CSR format Complete***************/


  /***********Now Call MKL Eigenvalue Solve Routine************/


/*
!   Content: Example for k Max/Min eigenvalue problem based on Intel MKL 
!            Extended Eigensolver (CSR sparse format, double precision)
!
!*******************************************************************************
!
! The following routines are used in the example:
!          MKL_SPARSE_D_EV
!
! Consider the 4x4 matrix A
!
!                 |  6   2   0   0   |
!                 |  2   3   0   0   |
!     A   =       |  0   0   2  -1   |
!                 |  0   0  -1   2   |
!
! stored as sparse matrix.
!
!
!  The test calls mkl_sparse_d_ev routine to find several largest singular 
!  values and corresponding right-singular vectors. Orthogonality of singular  
!  vectors is tested using DGEMM routine
!
!*******************************************************************************/

 /* Matrix A of size N in CSR format */
  MKL_INT N = num_cols;               /* number of rows in matrix A */
  MKL_INT M = num_cols;               /* number of columns in matrix A , since symmetric # cols = # rows*/
  MKL_INT nnz = ncc;             /* number of non-zeros in matrix */

 
//MKL_INT ia[5] = {1,3,5,7,9};                         /* ia array from CSR format */
//MKL_INT ja[8] = {1,2,1,2,3,4,3,4};                   /* ja array from CSR format */
//double   a[8] = {6.0,2.0,2.0,3.0,2.0,-1.0,-1.0,2.0}; /* val array from CSR format */

//double   Eig[4] = {1.0, 2.0, 3.0, 7.0}; /* Exact eigenvalues */

/* mkl_sparse_d_ev input parameters */
  char         which = 'L'; /* Which eigenvalues to calculate. ('L' - largest (algebraic) eigenvalues, 'S' - smallest (algebraic) eigenvalues) */
  MKL_INT      pm[128];     /* This array is used to pass various parameters to Extended Eigensolver Extensions routines. */
  MKL_INT      k0  = 5;     /* Desired number of max/min eigenvalues */   
  MKL_INT      k1  = k0*num_cols;  /* array size for Eigenvalues */

  /* mkl_sparse_d_ev output parameters */        
  MKL_INT      kk;           /* Number of eigenvalues found (might be less than k0). */    
  double       E[k0];        /* Eigenvalues */
  double       X[k1];        /* Eigenvectors */
  double       res[k0];      /* Residual */
 
  /* Local variables */
  MKL_INT      info;               /* Errors */
  MKL_INT      compute_vectors = 1;/* Flag to compute eigenvectors */
  MKL_INT      tol = 7;            /* Tolerance */
  double       Y[k0];               /* Y=(X')*X-I */
  double       sparsity;           /* Sparsity of randomly generated matrix */
  MKL_INT      i1, j1;
  double       smax, t;    
    
  /* Sparse BLAS IE variables */
  sparse_status_t status1;
  sparse_matrix_t AA = NULL; /* Handle containing sparse matrix in internal data structure */
  struct matrix_descr descr; /* Structure specifying sparse matrix properties */

  /* Create handle for matrix A stored in CSR format */
  descr.type = SPARSE_MATRIX_TYPE_GENERAL; /* Full matrix is stored */
  status1 = mkl_sparse_d_create_csr ( &AA, SPARSE_INDEX_BASE_ONE, N, N, ia, ia+1, ja, acsr );

  /* Step 2. Call mkl_sparse_ee_init to define default input values */
  mkl_sparse_ee_init(pm);

  pm[1] = tol; /* Set tolerance */
  pm[6] = compute_vectors; /* Compute Eigenvectors */

  /* Step 3. Solve the standard Ax = ex eigenvalue problem. */
  info = mkl_sparse_d_ev(&which, pm, AA, descr, k0, &kk, E, X, res);   

//  printf("mkl_sparse_d_ev output info %d \n",info);
//  if ( info != 0 )
//  {
//      printf("Routine mkl_sparse_d_ev returns code of ERROR: %i", (int)info);
//      return 1;
//  }

 /* printf("*************************************************\n");
  printf("************** REPORT ***************************\n");
  printf("*************************************************\n");
  printf("#mode found/subspace %d %d \n", kk, k0);
  printf("Index/Exact Eigenvalues/Residuals\n");
 

  for (i1=0; i1<kk; i1++)
  {
     printf("   %d  %.15e %.15e \n" ,i1, E[i1], res[i1]);
  }
 */
  for (int r = 0; r < kk ; r++){       // scaling eigenvalues 
 	  E[r] = E[r] * 1e-20;
  }

  mkl_sparse_destroy(AA);  // free memory for csr matrix handle

/*******Eigen value/vector Computation Complete *********/


/****Coarse Grid Matrix = Vector of Eigenvalues AND Prolongation Matrix = Eigenvectors****/


//   Construct Smoother: D = blkdiag(A(1:nJ1, 1:nJ1), A(nJ+1:n, nJ+1:n))

/**********Domain Decomposition************/


  //Allocating memory for blocks
  //MSC->p = new int[sizeG+1];
  D->p = new int[sizeD+1];
  L->p = new int[sizeD+1];
  G->p = new int[sizeG+1];  
  U->p = new int[sizeG+1];
  B->p = new int[sizeB+1];


  //MSC->nz = -1;MSC->m = sizeG;MSC->n = sizeG;
  D->nz = -1;D->m = sizeD;D->n = sizeD;
  G->nz = -1;G->m = sizeG;G->n = sizeG;
  L->nz = -1;L->m = sizeG;L->n = sizeD;
  U->nz = -1;U->m = sizeD;U->n = sizeG;
  B->nz = -1;B->m = sizeB;B->n = sizeB;  


  //cout << "\nCounting non zeros ...\n";
  for(j=0;j<sizeD;j++)
  {
    for(k=A->p[j]; k < A->p[j+1]; k++)
    {
      if(A->i[k] < sizeD){ 
        ++nzD;
        //++nzB;
      }
      else ++nzL;
    }
  }

  
  nzG = ncc - (nzD + 2*nzL); 
  nzB = nzD + nzG;

  //Allocating memory
  D->i = new int[nzD];
  D->x = new double[nzD];
  L->i = new int[nzL];
  L->x = new double[nzL];
  U->i = new int[nzL];
  U->x = new double[nzL];
  G->i = new int[nzG];
  G->x = new double[nzG];
  B->i = new int[nzB];   // num of cols = nzB
  B->x = new double[nzB];   // nzB = nzD + nzG

  //MSC->i = new int[sizeG*sizeG];
  //MSC->x = new double[sizeG*sizeG];

  //setting values
  D->nzmax = nzD;
  L->nzmax = nzL; 
  U->nzmax = nzL;
  G->nzmax = nzG;  
  B->nzmax = nzB;
  //MSC->nzmax = sizeG*sizeG;

  
 // cout << "\nFilling non zeros ...\n";
  for(j=0;j<sizeD;j++)
  {
    D->p[j] = iterD;
    L->p[j] = iterL;
    B->p[j] = iterD;

    for(k=A->p[j]; k < A->p[j+1]; k++)
    {
      if(A->i[k] < sizeD) 
      {
        D->i[iterD] = A->i[k];
        D->x[iterD] = A->x[k];
        B->i[iterD] = A->i[k];     // Filling B block upto D blocks
        B->x[iterD] = A->x[k];
        iterD += 1;
        iterB += 1;
        //++nzD_col;
      }
      else
      {
        L->i[iterL] = A->i[k] - sizeD;
        L->x[iterL] = A->x[k];
        iterL += 1;
      }
    }
  }
  D->p[sizeD] = iterD;
 // B->p[sizeD] = iterB;    // Doubt !!!, iterB = iterD
  L->p[sizeD] = iterL;
  
  status = umfpack_di_transpose(sizeG,sizeD, L->p,L->i,L->x,null,null,U->p,U->i,U->x) ;
  //cout << "\n TRANSPOSE STATUS : "<< status<< "\n";
  

  for(j = sizeD;j<num_cols;j++)
  {
    G->p[j - sizeD] = iterG;
    B->p[j] = iterB;            // as B already filled upto D
    if(j < num_cols-1)
    {
      //cout << "\n In first condition...\n";
      for(k = A->p[j]; k < A->p[j+1];k++)
      {
        if(A->i[k] < sizeD) continue;
        else
        {
          //cout << "\n row : " << A->i[k] << "\n";
          G->i[iterG] = A->i[k]-sizeD;
          G->x[iterG] = A->x[k];
          B->i[iterB] = A->i[k];     // Don't subtract sizeD
          B->x[iterB] = A->x[k];
          iterG += 1;
          iterB += 1;
        }     
      } 
    }
    //for the last column
    else 
    {
      //cout << "\n In second condition..."<<A->nzmax <<"\n";
      for(k = A->p[j]; k < A->nzmax;k++)
      {
        if(A->i[k] < sizeD) continue;
        else
        {
          //cout << "\n row : " << A->i[k] << "\n";
          G->i[iterG] = A->i[k] - sizeD;
          G->x[iterG] = A->x[k];
          B->i[iterB] = A->i[k];     // Don't Subtract sizeD
          B->x[iterB] = A->x[k];
          iterG += 1;
          iterB += 1;
        }
        //cout << "\n k : " << k << "\n";
        //break;
      }
    }
      //break;
  }

  G->p[sizeG] = iterG;
  B->p[sizeB] = iterB;    // iterB = iterD + iterG



delete [] G->p; delete [] G->i; delete [] G->x; delete G;
delete [] U->p;delete [] U->i;delete [] U->x;delete U;
 // cout << "\nFilling non zeros complete!!\n";

  
  /**********Convert B block to CSR****************/

  MKL_INT ivar1;
  char cvar1;
  ivar1 = sizeB;
  cvar1 = 'N'; //no transpose
  MKL_INT job1[6] = {1,1,0,0,0,1};         // various parameters for conversion
  double *bacsr = new double[B->nzmax];    // values
  MKL_INT *bja = new MKL_INT[B->nzmax];    // column
  MKL_INT *bia = new MKL_INT[sizeB];       // rows
  MKL_INT info2;                           // status

  // call to routine dcsrcsr
  mkl_dcsrcsc(job1,&ivar1,bacsr,bja,bia,B->x,B->i,B->p,&info2);
  //mkl_dcsrcsc(job,&ivar,acsr,ja,ia,A->x,A->i,A->p,&info1);

//  cout << "\nConversion info B : "<< info2 << endl;


/*************Matrix Conversion of B block in CSR format Complete!!!*************/

/**********Compute LU factorization of Smoother BLK-diagonal Matrix - B*********/

/*

  dcsrilu0 (const MKL_INT *n , const double *a , const MKL_INT *ia , const MKL_INT *ja , double *bilu0 , const MKL_INT *ipar , const double *dpar , MKL_INT *ierr );
*/

  // paramters for ilu routine
  MKL_INT ipar1[sizelu];
  double dpar1[sizelu];
  MKL_INT ierr1 = 0;
  double *bilu0 = new double[B->nzmax];

  // Initialize the parameters
  ipar1[30] = 1;
  dpar1[30] = 1.E-20;  // 1.0e-20
  dpar1[31] = 1.E-16;   //  1.0e-16

  // call to ILU routine for B
  dcsrilu0 (&ivar1, bacsr, bia, bja, bilu0, ipar1, dpar1, &ierr1);

//  printf ("\ndcsrilu0 has returned the code %d\n", ierr1);
//  printf("\n");


  /*******Approximate LU factorization complete using ILU**********/
  
//  cout << "Checking first few entries of D and B" << endl;
//  for (int l=0 ; l<5; l++)
//    cout << *(bacsr+l) << " -- " << D->x[l] << endl;
//    cout << endl;




/**********************GMRES 
							CALL
								******************************/

//initializing variables and data structures for DFGMRES call
	//int restart = 20;  //DFGMRES restarts
   MKL_INT* ipar = new MKL_INT[size_MKL_IPAR];
   //ipar[14] = 150;  //non restarted iterations
   
   //cout << "\n tmp size : "<< num_cols*(2*ipar[14]+1)+ipar[14]*((ipar[14]+9)/2+1) << "\n";

   double* dpar = new double[size_MKL_IPAR]; 
  
   //double tmp[num_cols*(2*ipar[14]+1)+ipar[14]*((ipar[14]+9)/2+1)];
   //double* tmp = new double[num_cols*(2*ipar[14]+1)+(ipar[14]*(ipar[14]+9))/2+1];
    
   double* tmp = new double[num_cols*(2*150+1)+(150*(150+9))/2+1];
  
   //double expected_solution[num_cols];
   
   double* rhs = new double[num_cols];
   double* computed_solution = new double[num_cols];
   double* residual = new double[num_cols];   
   double nrm2,rhs_nrm,relres_nrm,dvar,relres_prev,prec_rhs_nrm,prec_relres_nrm;
   double *prec_rhs = new double[num_cols];
	
   

   MKL_INT itercount,ierr=0;
   MKL_INT RCI_request, RCI_count;
  // char cvar;

 //  cout << "\nMKL var init done !\n";



 //  ivar = num_cols;
 //  cvar = 'N';  //no transpose
	
	/**********Converting A & L from CSC to CSR*****************/
   MKL_INT job6[6] = {1,1,0,0,0,1};
//   double *acsr =  new double[ncc];
   double *lcsr =  new double[L->nzmax];
//   MKL_INT *ja = new MKL_INT[ncc];
   MKL_INT *jl = new MKL_INT[L->nzmax];
//   MKL_INT *ia = new MKL_INT[ivar+1];
   MKL_INT *il = new MKL_INT[sizeG+1];
   MKL_INT info6;
   MKL_INT lvar = sizeG;

   //converting COO to CSR
 //  mkl_dcsrcsc(job,&ivar,acsr,ja,ia,A->x,A->i,A->p,&info);
   //cout << "\n Conversion info A : "<< info << "\n";


   mkl_dcsrcsc(job6,&lvar,lcsr,jl,il,L->x,L->i,L->p,&info6);
   
   //cout << "\n Conversion info L : "<< info << "\n";

   /***Testing the preconditioner solve with MATLAB***/
   //test_prec_solve(A,D,MSC,Numeric_D,Numeric_MSC,lcsr,il,jl);

   /*---------------------------------------------------------------------------
   /* Save the right-hand side in vector rhs for future use
   /*---------------------------------------------------------------------------*/
   RCI_count=1;
   /** extracting the norm of the rhs for computing rel res norm**/
   rhs_nrm = dnrm2(&ivar,Jt_e,&RCI_count); 
   //cout << "\n rhs_nrm : " << rhs_nrm << "\n";
   // JTe vector is not altered
   //rhs is used for residual calculations
  // dcopy(&ivar, JTe, &RCI_count, rhs, &RCI_count);   
   for(int q = 0; q < num_cols; q++) rhs[q] = Jt_e[q];

   // PRECONDITIONED RHS
// tg_solve(double *acsr,int *ia,int *ja, double *bilu0, double E[], double X[], double *y_in, double *z_out)
   tg_solve(num_cols, kk, acsr, ia, ja, bilu0, bia, bja, E, X, Jt_e, prec_rhs);

   
 //prec_solve(A,D,MSC,Numeric_D,Numeric_MSC,lcsr,il,jl,JTe,prec_rhs);
   //norm of the preconditioned rhs
   prec_rhs_nrm = dnrm2(&ivar,prec_rhs,&RCI_count);
   delete [] prec_rhs; 
//   cout << "\n prec rhs norm : "<< prec_rhs_nrm << "\n";

	/*---------------------------------------------------------------------------
	/* Initialize the initial guess
	/*---------------------------------------------------------------------------*/
	for(RCI_count=0; RCI_count<num_cols; RCI_count++)
	{
		computed_solution[RCI_count]=0.0;
	}
//  if(ipar[10] == 1) computed_solution[0]=1000.0;   // for preconditioner GMRES to work
	//computed_solution[0]=0.0;

	/*---------------------------------------------------------------------------
	/* Initialize the solver
	/*---------------------------------------------------------------------------*/
	dfgmres_init(&ivar, computed_solution, Jt_e, &RCI_request, ipar, dpar, tmp); 
	if (RCI_request!=0) goto FAILED;

	
	ipar[7] = 1;
	ipar[4] = 500;  // Max Iterations
	ipar[10] = 1;  //Preconditioner used
	ipar[14] = 20; //internal iterations
	
	dpar[0] = 1.0e-04; //Relative Tolerance

	/*---------------------------------------------------------------------------
	/* Check the correctness and consistency of the newly set parameters
	/*---------------------------------------------------------------------------*/
	dfgmres_check(&ivar, computed_solution, rhs, &RCI_request, ipar, dpar, tmp); 
	if (RCI_request!=0) goto FAILED;

	/*---------------------------------------------------------------------------
	/* Compute the solution by RCI (P)FGMRES solver with preconditioning
	/* Reverse Communication starts here
	/*---------------------------------------------------------------------------*/
	ONE: dfgmres(&ivar, computed_solution, Jt_e, &RCI_request, ipar, dpar, tmp);
//	cout << "\n dfgmres RCI_request : "<<RCI_request << "\n";

	//ipar[13] = 20; // No of inner iterations
	if(RCI_request==0) goto COMPLETE;

	/*---------------------------------------------------------------------------
	/* If RCI_request=1, then compute the vector A*tmp[ipar[21]-1]
	/* and put the result in vector tmp[ipar[22]-1]	
	/*------------------DEPRECATED ROUTINE (FIND ANOTHER )-------------------------*/
	if (RCI_request==1)
	{
		mkl_dcsrgemv(&cvar, &ivar, acsr, ia, ja, &tmp[ipar[21]-1], &tmp[ipar[22]-1]);
		goto ONE;
	}

	/*---------------------------------------------------------------------------
	/* If RCI_request=2, then do the user-defined stopping test
	/* The residual stopping test for the computed solution is performed here*/
	if (RCI_request==2)
	{
		/* Request to the dfgmres_get routine to put the solution into b[N] via ipar[12]*/
		ipar[12]=1;
		/* Get the current FGMRES solution in the vector rhs[N] */
		dfgmres_get(&ivar, computed_solution, rhs, &RCI_request, ipar, dpar, tmp, &itercount);

		//for(int kl = 0; kl < 10; kl++)
		//	printf("\n comp_sol[%d] = %10.9f",kl, computed_solution[kl]);

		/* Compute the current true residual via MKL (Sparse) BLAS routines */
		mkl_dcsrgemv(&cvar, &ivar, acsr, ia, ja, rhs, residual); // A*x for new solution x
		dvar=-1.0E0;
		RCI_count=1;
		daxpy(&ivar, &dvar,  Jt_e, &RCI_count, residual, &RCI_count);  // Ax - A*x_solution
		

	//	cout << "\n Iteration : " << itercount;
		//for(int k = 0; k < 10; k++)
		//		printf("\nresildual[%d] = %10.9f\n",k,residual[k]);

		if(ipar[10] == 0)   // non preconditioned system
		{
			dvar=dnrm2(&ivar,residual,&RCI_count);
			relres_nrm = dvar/rhs_nrm;
			//relres_nrm = dvar/prec_rhs_nrm;
			//printf("\nresidual norm non prec = %10.9f\n",dvar);
			
		}
		else if(ipar[10] == 1)  //preconditioned system
		{
			double *prec_relres = new double[num_cols];
			//dvar=dnrm2(&ivar,residual,&RCI_count);
			//printf("\nresidual norm with prec = %10.9f\n",dvar);
			tg_solve(num_cols, kk, acsr, ia, ja, bilu0, bia, bja, E, X, residual, prec_relres);
		//	prec_solve(A,D,MSC,Numeric_D,Numeric_MSC,lcsr,il,jl,residual,prec_relres);

			prec_relres_nrm = dnrm2(&ivar,prec_relres,&RCI_count); 
			delete [] prec_relres;
			//cout << "\n prec relres norm : " << prec_relres_nrm << "\n";
			//printf("\nPrec relres norm : %10.9f",prec_relres_nrm);
			relres_nrm = prec_relres_nrm/prec_rhs_nrm;

		}

		//cout << "\n relres_nrm : " << relres_nrm << "\n";
//		printf("\nRelres norm = %10.9f\n",relres_nrm);

		if (relres_nrm<1.0E-2) goto COMPLETE;   //taking tolerance as 1e-04

		else goto ONE;
		
	}

	/*---------------------------------------------------------------------------
	/* If RCI_request=3, then apply the preconditioner on the vector
	/* tmp[ipar[21]-1] and put the result in vector tmp[ipar[22]-1]*/
	
	if (RCI_request==3)
	{
		//cout << "\n Prec solve ..." << "\n";
		tg_solve(num_cols, kk, acsr, ia, ja, bilu0, bia, bja, E, X, &tmp[(ipar[21]-1)], &tmp[(ipar[22]-1)]);
//		prec_solve(A,D,MSC,Numeric_D,Numeric_MSC,lcsr,il,jl,&tmp[(ipar[21]-1)],&tmp[(ipar[22]-1)]);
		goto ONE;
	}

	/*---------------------------------------------------------------------------
	/* If RCI_request=4, then check if the norm of the next generated vector is
	/* not zero up to rounding and computational errors. The norm is contained
	/* in dpar[6] parameter
	/*---------------------------------------------------------------------------*/
	if (RCI_request==4)
	{
		if (dpar[6]<1.0E-2) goto COMPLETE;
		else goto ONE;
	}
	/*---------------------------------------------------------------------------
	/* If RCI_request=anything else, then dfgmres subroutine failed
	/* to compute the solution vector: computed_solution[N]
	/*---------------------------------------------------------------------------*/
	else
	{
		goto FAILED;
	}
	/*---------------------------------------------------------------------------
	/* Reverse Communication ends here
	/* Get the current iteration number and the FGMRES solution (DO NOT FORGET to
	/* call dfgmres_get routine as computed_solution is still containing
	/* the initial guess!). Request to dfgmres_get to put the solution
	/* into vector computed_solution[N] via ipar[12]
	/*---------------------------------------------------------------------------*/
	COMPLETE:   ipar[12]=0;
	dfgmres_get(&ivar, computed_solution, Jt_e, &RCI_request, ipar, dpar, tmp, &itercount);
//	cout << "The system has been solved  in " << itercount << " iterations!\n";
//	cout << "\n RCI_request : "<< RCI_request << "\n";
	RCI_count = 1; 

  for(int q = 0; q < num_cols; q++) delta[q] = computed_solution[q];
//	cout << "Norm of solution = " << dnrm2(&ivar, computed_solution, &RCI_count) << endl;
//	printf("\nThe following solution has been obtained: \n");

//	for (RCI_count=0;RCI_count<10;RCI_count++)                //PRINTING ONLY THE FIRST 10 MEMBERS
//	{
//		printf("computed_solution[%d]=%10.12f\n",RCI_count,computed_solution[RCI_count]);
//	}

	MKL_Free_Buffers();


	//  Free the numeric factorization.
//  umfpack_di_free_numeric ( &Numeric_D );
//  umfpack_di_free_numeric ( &Numeric_MSC );

//	delete [] JTe; 
	delete [] A->p; delete [] A->i; delete [] A->x; 
//	delete [] MSC->p; delete [] MSC->i;delete [] MSC->x;
	delete [] D->p;  delete [] D->i;delete [] D->x; 
	delete [] L->p; delete [] L->i; delete [] L->x;
	delete [] B->p; delete [] B->i; delete [] B->x;
//	delete MSC;
	delete D; delete A;  delete L; delete B;

	delete [] tmp; delete [] ipar; delete [] dpar;
	delete [] rhs; delete [] computed_solution; delete [] residual;
	delete [] acsr; delete [] ia; delete [] ja;
	delete [] lcsr; delete [] il; delete [] jl;
	delete [] bacsr; delete [] bia; delete [] bja;

	return ;

	FAILED: cout << "The solver has returned the ERROR code " << RCI_request << "\n";

	MKL_Free_Buffers();

	return ;

    NOT_CONVERGE: cout << "The relative residual did not change for successive iterations !\n";

  return ;

  }


}
#endif
