// Problem size
# define N 512

// 0 ... input matrix not transposed
// 1 ... input matrix transposed
# define TRANSPOSED 0 

__kernel void matrix_mult(__global const float *A,
                        __global const float *B,
                        __global float *restrict C)
{
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);

  float sum =  0.0;

  for(unsigned i = 0; i<N; ++i) {
  # if TRANSPOSED == 0
    sum += A[N*x + i]*B[N*i + y]; 
  # endif

  # if TRANSPOSED == 1
    sum += A[N*x + i]*B[N*y + i];
  # endif
  }
  C[N*x + y] = sum;
}
