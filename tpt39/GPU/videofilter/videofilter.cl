#define N 640
#define M 360


/*
https://en.wikipedia.org/wiki/Kernel_(image_processing)

kernelGauss = 1/16 * {1,2,1,
                      2,4,2,
                      1,2,1}
	

char kernelSobel = {-1,-1,-1,
                    -1, 8,-1,
                    -1, 1,-1};
*/

__kernel void tunnel(__global const unsigned char *I,
                     __global unsigned char * restrict O)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    O[x + N*y] = I[x + N*y];
}

__kernel void gauss(__global const unsigned char *I,
                    __global unsigned char * restrict O)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    unsigned short sum = 0;

    // Convolution
    if(x != 0 && y != 0 && x != N-1 && y != M-1) {
        sum += I[N * (y-1) + x-1];
        sum += I[N * (y-1) + x  ] * 2;
        sum += I[N * (y-1) + x+1];
        sum += I[N * (y)   + x-1] * 2;
        sum += I[N * (y)   + x  ] * 4;
        sum += I[N * (y)   + x+1] * 2;
        sum += I[N * (y+1) + x-1];
        sum += I[N * (y+1) + x  ] * 2;
        sum += I[N * (y+1) + x+1];
    }  
    O[x + N*y] = (unsigned char)(sum/16);
}

__kernel void sobel(__global const unsigned char *I,
                    __global unsigned char * restrict O)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    unsigned char sum, tmp = 0;

    // Convolution
    if(x != 0 && y != 0 && x != N-1 && y != M-1) {
        
        tmp = I[N * (y)   + x  ] * 8;
        sum += tmp;
        sum -= I[N * (y-1) + x-1];
        sum -= I[N * (y-1) + x  ];
        sum -= I[N * (y-1) + x+1];
        sum -= I[N * (y)   + x-1];
        sum -= I[N * (y)   + x+1];
        sum -= I[N * (y+1) + x-1];
        sum -= I[N * (y+1) + x  ];
        sum -= I[N * (y+1) + x+1];
        
        if(sum > tmp) {
            sum = 0;
        }
    } 
  
    O[x + N*y] = sum;
}
