/* Test and timing harness program for developing a dense matrix
   multiplication routine for the CS3014 module */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>

/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
/*#define DEBUGGING(_x) _x */
/* to stop the printing of debugging information, use the following line: */
#define DEBUGGING(_x)


/* write matrix to stdout */
void write_out(double ** a, int dim1, int dim2)
{
  int i, j;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2 - 1; j++ ) {
      printf("%f, ", a[i][j]); //change to f
    }
    printf("%f\n", a[i][dim2-1]);
  }
}


/* create new empty matrix */
double ** new_empty_matrix(int dim1, int dim2)
{
  double ** result = malloc(sizeof(double*) * dim1);
  double * new_matrix = malloc(sizeof(double) * dim1 * dim2);
  int i;

  for ( i = 0; i < dim1; i++ ) {
    result[i] = &(new_matrix[i*dim2]);
  }

  return result;
}

/* take a copy of the matrix and return in a newly allocated matrix */
double ** copy_matrix(double ** source_matrix, int dim1, int dim2)
{
  int i, j;
  double ** result = new_empty_matrix(dim1, dim2);

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      result[i][j] = source_matrix[i][j];
    }
  }

  return result;
}

/* create a matrix and fill it with random numbers */
double ** gen_random_matrix(int dim1, int dim2)
{
  double ** result;
  int i, j;
  struct timeval seedtime;
  int seed;

  result = new_empty_matrix(dim1, dim2);

  /* use the microsecond part of the current time as a pseudorandom seed */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /* fill the matrix with random numbers */
  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      long long upper = random();
      long long lower = random();
      result[i][j] = (double)((upper << 32) | lower);
    }
  }

  return result;
}

/* check the sum of absolute differences is within reasonable epsilon */
void check_result(double ** result, double ** control, int dim1, int dim2)
{
  int i, j;
  double sum_abs_diff = 0.0;
  const double EPSILON = 0.0625;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      double diff = abs(control[i][j] - result[i][j]);
      sum_abs_diff = sum_abs_diff + diff;
    }
  }

  if ( sum_abs_diff > EPSILON ) {
    fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
      sum_abs_diff, EPSILON);
  } else {
    printf("Result Correct!\n");
  }
}

/* multiply matrix A times matrix B and put result in matrix C */
void matmul(double ** A, double ** B, double ** C, int a_dim1, int a_dim2, int b_dim2)
{
  int i, j, k;

  for ( i = 0; i < a_dim1; i++ ) {
    for( j = 0; j < b_dim2; j++ ) {
      double sum = 0.0;
      for ( k = 0; k < a_dim2; k++ ) {
  sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

/* the fast version of matmul written by the team */
void team_matmul(double ** A, double ** B, double ** C, int a_dim1, int a_dim2, int b_dim2)
{
  int work = a_dim1 * a_dim2;

  if (work < 6500) {
    matmul(A, B, C, a_dim1, a_dim2, b_dim2); // don't parrallel
    return;
  }
  int i, j, k;
  double sum;

#pragma omp parallel for private(i, j, k, sum)
  for ( i = 0; i < a_dim1; i++ ) {
    // don't put anopther parallel for here, one is better
    for( j = 0; j < b_dim2-1; j+=1 ) { // loop unrolling
      sum = 0.0;
      for ( k = 0; k < a_dim2; k++ ) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;

      j += 1;
      sum = 0.0;
      for ( k = 0; k < a_dim2; k++ ) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
    for(; j < b_dim2; j+=1 ) {
      sum = 0.0;
      for ( k = 0; k < a_dim2; k++ ) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }

}

unsigned long upper_power_of_two(unsigned long v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

void matadd(double ** A, double ** B, double ** C, int dim)
{
  int i,j;
  for(i = 0; i < dim; i++)
  {
    for(j = 0; j < dim; j++)
    {
      C[i][j] = A[i][j] + B[i][j];
    }
  }
}

void matsub(double ** A, double ** B, double ** C, int dim)
{
  int i,j;
  for(i = 0; i < dim; i++)
  {
    for(j = 0; j < dim; j++)
    {
      C[i][j] = A[i][j] - B[i][j];
    }
  }
}

void matquart(double ** A, double ** A11, double ** A12, double ** A21, double ** A22, int dim)
{
  int dim2 = dim / 2;
  int i,j;
  #pragma omp parallel for private(i, j)
    for(i = 0; i < dim2; i++)
    {
      for(j = 0; j < dim2; j++)
      {
        A11[i][j] = A[i][j];
        A12[i][j] = A[i][j+dim2];
        A21[i][j] = A[i+dim2][j];
        A22[i][j] = A[i+dim2][j+dim2];
      }
    }

}

void matunquart(double ** A, double ** A11, double ** A12, double ** A21, double ** A22, int dim)
{
  int dim2 = dim / 2;
  int i,j;
  #pragma omp parallel for private(i, j)
    for(i = 0; i < dim2; i++)
    {
      for(j = 0; j < dim2; j++)
      {
        A[i][j]           = A11[i][j];
        A[i][j+dim2]      = A12[i][j];
        A[i+dim2][j]      = A21[i][j];
        A[i+dim2][j+dim2] = A22[i][j];
      }
    }

}
void strassen(double ** A, double ** B, double ** C, int n)
{
  fflush(stdout);
  // just testing whether to do ordinary stuff
  if (n <= 1024) {
    team_matmul(A,B,C,n,n,n);
    return;
  }

  int new_n = n/2;

  double **M1, **M2, **M3, **M4, **M5, **M6, **M7, 
  **A11, **A12, **A21, **A22, **B11, **B12, **B21, 
  **B22, **C11, **C12, **C21, **C22;

  double **temp1, **temp2, **temp3, **temp4, **temp5
  , **temp6, **temp7, **temp8, **temp9, **temp10, **temp11, **temp12;
  M1 = new_empty_matrix(new_n, new_n);
  M2 = new_empty_matrix(new_n, new_n);
  M3 = new_empty_matrix(new_n, new_n);
  M4 = new_empty_matrix(new_n, new_n);
  M5 = new_empty_matrix(new_n, new_n);
  M6 = new_empty_matrix(new_n, new_n);
  M7 = new_empty_matrix(new_n, new_n);

  // this is starting to look bad...

  // but it's only going to get better!

  A11 = new_empty_matrix(new_n, new_n);
  A12 = new_empty_matrix(new_n, new_n);
  A21 = new_empty_matrix(new_n, new_n);
  A22 = new_empty_matrix(new_n, new_n);
  B11 = new_empty_matrix(new_n, new_n);
  B12 = new_empty_matrix(new_n, new_n);
  B21 = new_empty_matrix(new_n, new_n);
  B22 = new_empty_matrix(new_n, new_n);


  matquart(A,A11,A12,A21,A22,n);


  
  matquart(B,B11,B12,B21,B22,n);

  fflush(stdout);

  #pragma omp parallel sections
  {
    #pragma omp section
    {
      temp1 = new_empty_matrix(new_n, new_n);
      temp2 = new_empty_matrix(new_n, new_n);
      matadd(A11, A22, temp1, new_n);
      matadd(B11, B22, temp2, new_n);
      strassen(temp1, temp2, M1, new_n);
      free(temp1);
      free(temp2);
    }

    #pragma omp section
    {    
      temp3 = new_empty_matrix(new_n, new_n);
      temp4 = new_empty_matrix(new_n, new_n);
      matadd(A21, A22, temp3, new_n);
      strassen(temp3, B11, M2, new_n);
      free(temp3);
    }

    #pragma omp section
    {
      temp4 = new_empty_matrix(new_n, new_n);
      matsub(B12, B22, temp4, new_n);
      strassen(A11, temp4, M3, new_n);
      free(temp4);
    }

    #pragma omp section
    {
      temp5 = new_empty_matrix(new_n, new_n);
      matsub(B21, B11, temp5, new_n);
      strassen(A22, temp5, M4, new_n);
      free(temp5);
    }

    #pragma omp section
    {
      temp6 = new_empty_matrix(new_n, new_n);
      matadd(A11, A12, temp6, new_n);
      strassen(B22, temp6, M5, new_n);
      free(temp6);
    }

    #pragma omp section
    {
      temp7 = new_empty_matrix(new_n, new_n);
      temp8 = new_empty_matrix(new_n, new_n);
      matsub(A21, A11, temp7, new_n);
      matadd(B11, B12, temp8, new_n);
      strassen(temp7, temp8, M6, new_n);
      free(temp7);
      free(temp8);
    }

    #pragma omp section
    {
      temp9 = new_empty_matrix(new_n, new_n);
      temp10 = new_empty_matrix(new_n, new_n);
      matsub(A12, A22, temp9, new_n);
      matadd(B21, B22, temp10, new_n);
      strassen(temp9, temp10, M7, new_n);
      free(temp9);
      free(temp10);
    }

  }


  temp11 = new_empty_matrix(new_n, new_n);
  temp12 = new_empty_matrix(new_n, new_n);



  C11 = new_empty_matrix(new_n, new_n);
  C12 = new_empty_matrix(new_n, new_n);
  C21 = new_empty_matrix(new_n, new_n);
  C22 = new_empty_matrix(new_n, new_n);

  matadd(M1, M4, temp11, new_n);
  matsub(temp11, M5, temp12, new_n);
  matadd(temp12, M7, C11, new_n);



  matadd(M3, M5, C12, new_n);

  matadd(M2, M4, C21, new_n);

  matsub(M1, M2, temp11, new_n);
  matadd(temp11, M3, temp12, new_n);
  matadd(temp12, M6, C22, new_n);

  free(temp11);
  free(temp12);

  matunquart(C,C11,C12,C21,C22,n);
  free(C11);
  free(C12);
  free(C21);
  free(C22);

}

int main(int argc, char ** argv)
{
  double ** A, ** B, ** C;
  double ** control_matrix;
  long long mul_time;
  int a_dim1, a_dim2, b_dim1, b_dim2;
  struct timeval start_time;
  struct timeval stop_time;

  if ( argc != 5 ) {
    fprintf(stderr, "Usage: matmul-harness <A nrows> <A ncols> <B nrows> <B ncols>\n");
    exit(1);
  }
  else {
    a_dim1 = atoi(argv[1]);
    a_dim2 = atoi(argv[2]);
    b_dim1 = atoi(argv[3]);
    b_dim2 = atoi(argv[4]);
  }

  /* check the matrix sizes are compatible */
  if ( a_dim2 != b_dim1 ) {
    fprintf(stderr,
      "FATAL number of columns of A (%d) does not match number of rows of B (%d)\n",
      a_dim2, b_dim1);
    exit(1);
  }

  /* allocate the matrices */
  A = gen_random_matrix(a_dim1, a_dim2);
  B = gen_random_matrix(b_dim1, b_dim2);
  C = new_empty_matrix(a_dim1, b_dim2);
  control_matrix = new_empty_matrix(a_dim1, b_dim2);

  DEBUGGING(write_out(A, a_dim1, a_dim2));


  /* record starting time */
  gettimeofday(&start_time, NULL);

  /* use a simple matmul routine to produce control result */
  matmul(A, B, control_matrix, a_dim1, a_dim2, b_dim2);

  /* record finishing time */
  gettimeofday(&stop_time, NULL);
  mul_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
    (stop_time.tv_usec - start_time.tv_usec);
  printf("Non-parrallel Matmul time: %lld microseconds\n", mul_time);


  /* record starting time */
  gettimeofday(&start_time, NULL);

  /* perform matrix multiplication */
  //team_matmul(A, B, C, a_dim1, a_dim2, b_dim2);
  strassen(A,B,C, a_dim1);

  /* record finishing time */
  gettimeofday(&stop_time, NULL);
  mul_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
    (stop_time.tv_usec - start_time.tv_usec);
  printf("Matmul time: %lld microseconds\n", mul_time);

  DEBUGGING(write_out(C, a_dim1, b_dim2));

  /* now check that the team's matmul routine gives the same answer
     as the known working version */
  check_result(C, control_matrix, a_dim1, b_dim2);

  return 0;
}
