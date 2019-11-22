#define N 640
#define M 360

/*
Used Convolutional Kernels

https://en.wikipedia.org/wiki/Kernel_(image_processing)

kernelGauss = 1/16 * {1,2,1,
                      2,4,2,
                      1,2,1}

https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
kernelScharX =       { -3,  0,  3,
                      -10,  0,  3,
                      -3,  0,  3}

kernelScharY =       { -3,-10, -3,
                        0,  0,  0,
                        3, 10,  3}

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
    // only filter non-edge pixels
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

    short sumX, sumY, out, thr = 0;

    // Convolution
    if(x != 0 && y != 0 && x != N-1 && y != M-1) {
        
        sumX += I[N * (y-1) + x+1] * 3;
        sumX += I[N * (y)   + x+1] * 10;
        sumX += I[N * (y+1) + x+1] * 3;

        sumX -= I[N * (y-1) + x-1] * 3;
        sumX -= I[N * (y)   + x-1] * 10;
        sumX -= I[N * (y+1) + x-1] * 3;

        sumY += I[N * (y+1) + x-1] * 3;
        sumY += I[N * (y+1) + x  ] * 10;
        sumY += I[N * (y+1) + x+1] * 3;
        
        sumY -= I[N * (y-1) + x-1] * 3;
        sumY -= I[N * (y-1) + x  ] * 10;
        sumY -= I[N * (y-1) + x+1] * 3;
    }

    if(sumX<0) sumX = 0;
    if(sumY<0) sumY = 0;
  
    thr = (sumX+sumY);

    if(thr < 100) {
        out = 255;
    } else {
        out = I[N*y+x];
    }

    O[x + N*y] = (unsigned char) out;
}

// does the calcuatios with floats
__kernel void sobelHR(__global const unsigned char *I,
                    __global unsigned char * restrict O)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    float sumX, sumY, thr, out = 0.0;

    // Convolution
    if(x != 0 && y != 0 && x != N-1 && y != M-1) {
        
        sumX += I[N * (y-1) + x+1] * 3;
        sumX += I[N * (y)   + x+1] * 10;
        sumX += I[N * (y+1) + x+1] * 3;

        sumX -= I[N * (y-1) + x-1] * 3;
        sumX -= I[N * (y)   + x-1] * 10;
        sumX -= I[N * (y+1) + x-1] * 3;

        sumY += I[N * (y+1) + x-1] * 3;
        sumY += I[N * (y+1) + x  ] * 10;
        sumY += I[N * (y+1) + x+1] * 3;
        
        sumY -= I[N * (y-1) + x-1] * 3;
        sumY -= I[N * (y-1) + x  ] * 10;
        sumY -= I[N * (y-1) + x+1] * 3;
    }

    if(sumX<0) sumX = 0;
    if(sumY<0) sumY = 0;
  
    thr = sqrt(pow(sumX,2) + pow(sumY,2));

    if(thr < 100) {
        out = 255;
    } else {
        out = I[N*y+x];
    }

    O[x + N*y] = (unsigned char) out;
}

