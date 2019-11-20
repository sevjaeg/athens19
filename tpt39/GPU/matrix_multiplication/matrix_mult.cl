#define N 512

__kernel void matrix_mult(__global const float *A,
                        __global const float *B,
                        __global float *restrict C)
{
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);

  float sum =  0.0;

  for(unsigned i = 0; i<N; ++i) {
    sum += A[N*x + i]*B[N*i + y]; 

    // alternatively transposed
    //sum += A[N*x + i]*B[N*y + i]; 
  }

  C[N*x + y] = sum;
}
