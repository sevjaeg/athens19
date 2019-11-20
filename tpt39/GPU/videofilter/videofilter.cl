#define N 640
#define M 360

__kernel void filter(__global const char *I,
                        __global const char *F,
                        __global const int *FACTOR,
                        __global float char * O)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    char sum = 0;

    // Convolution
    sum += I[N * (y-1) + x-1] * F[0];
    sum += I[N * (y-1) + x  ] * F[1];
    sum += I[N * (y-1) + x+1] * F[2];
    sum += I[N * (y)   + x-1] * F[3];
    sum += I[N * (y)   + x  ] * F[4];
    sum += I[N * (y)   + x+1] * F[5];
    sum += I[N * (y+1) + x-1] * F[6];
    sum += I[N * (y+1) + x  ] * F[7];
    sum += I[N * (y+1) + x+1] * F[8];

    O[x + N*y] = sum/FACTOR;
}
