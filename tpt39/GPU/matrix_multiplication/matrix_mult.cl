__kernel void matrix_mult(__global const float *A,
                        __global const float *B,
                        __global float *restrict C)
{
  const unsigned N = 1000;

  size_t x = get_global_id(0);
  size_t y = get_global_id(1);

  C[N*x + y] = 0.0;

  for(unsigned i = 0; i<N; ++i) {
    //Preferably the other way around for B (cache)
    C[N*x + y] += A[N*x + i]*B[N*i + y]; 
  }
}
