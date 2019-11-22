#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <fstream>
#include <time.h>
#include "opencv2/opencv.hpp"

#include <CL/cl.h>
#include <CL/cl_ext.h>

using namespace cv;
using namespace std;

// FILTERTYPE
// 0 ... none
// 1 ... triple gauß
// 2 ... sobel
// 3 ... triple gauß and sobel
#define FILTERTYPE 2
#define N_FRAMES 300

void copyArray(char* input, int len,  char* output) {
	for(int i = 0; i <len; ++i) {
		output[i] = input [i];
	}
}

// TODO introduce some error handling
// Code adapted from https://gist.github.com/courtneyfaulkner/7919509
cl_platform_id findPlatform() {
    cl_uint i, j;
    char* info;
    size_t infoSize;

    cl_uint platformCount;
    cl_platform_id* platforms;

    //Extensions are commented out (change array sizes to 5)
    const char* attributeNames[4] = { "Name", "Vendor",
        "Version", "Profile"}; //, "Extensions" };
    const cl_platform_info attributeTypes[4] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
        CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE};
        //, CL_PLATFORM_EXTENSIONS };
    const int attributeCount = sizeof(attributeNames) / sizeof(char*);

    //get platform count
    clGetPlatformIDs(5, NULL, &platformCount);

    // get all platforms
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    // for each platform print all attributes
    for (i = 0; i < platformCount; i++) {
        printf("\n %d. Platform \n", i+1);
        for (j = 0; j < attributeCount; j++) {

            // get platform attribute value size
            clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
            info = (char*) malloc(infoSize);
            
            // get platform attribute value
            clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);

            printf("  %d.%d %-11s: %s\n", i+1, j+1, attributeNames[j], info);
            free(info);
        }
        printf("\n");
    }
    return platforms[0];
}

// Code adapted from https://gist.github.com/courtneyfaulkner/7919509
cl_device_id obtainDevice(cl_platform_id* platforms, cl_uint platformCount, int deviceIndex) {
    cl_uint i, j;
    char* value;
    size_t valueSize;
    cl_uint maxComputeUnits;

    cl_uint deviceCount;
    cl_device_id* devices = NULL;
    cl_device_id deviceToUse;


    for (i = 0; i < platformCount; i++) {
        cl_device_type devices_to_search_for = CL_DEVICE_TYPE_GPU; //ALL

        // get all devices
        clGetDeviceIDs(platforms[i], devices_to_search_for, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], devices_to_search_for, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++) {

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("%d. Device: %s\n", j+1, value);
            free(value);

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %d.%d Hardware version: %s\n", j+1, 1, value);
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf(" %d.%d Software version: %s\n", j+1, 2, value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);
        }

        // Device specific code here

        // Assumtion from here on: use just one device
        
    }
        deviceToUse = devices[deviceIndex];
        free(devices);
        return deviceToUse;
}

void callback(const char *buffer, size_t length, size_t final, void *user_data) {
     fwrite(buffer, 1, length, stdout);
}

void print_clbuild_errors(cl_program program,cl_device_id device) {
    cout<<"Program Build failed\n";
    size_t length;
    char buffer[2048];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
    cout<<"--- Build log ---\n "<<buffer<<endl;
    exit(1);
}

char ** read_file(const char *name) {
    size_t size;
    char **output = (char **)malloc(sizeof(char *));
    FILE* fp = fopen(name, "rb");
    if (!fp) {
        printf("no such file:%s",name);
        exit(-1);
    }

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    *output = (char *)malloc(size);
    char **outputstr=(char **)malloc(sizeof(char *));
    *outputstr= (char *)malloc(size);
    if (!*output) {
        fclose(fp);
        printf("mem allocate failure:%s",name);
        exit(-1);
    }

    if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
    fclose(fp);

    snprintf((char *)*outputstr,size,"%s\n",*output);
    return outputstr;
}

void checkError(int status, const char *msg) {
    if(status!=CL_SUCCESS) {
        printf("%s\n",msg);
    }
}

int main(int, char**)
{
	const int RESX = 640;
	const int RESY = 360;

	unsigned char* inputFrame;
	unsigned char* filteredFrame = (unsigned char*) malloc(RESY*RESX*sizeof(char));

	// OpenCL setup
	cl_platform_id platformToUse;
	cl_device_id device;
	cl_context context;
	cl_context_properties context_properties[] =
	{
		CL_CONTEXT_PLATFORM, 0,
		CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
		CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
		0
	};
	cl_command_queue queue;
	cl_program program;

#if FILTERTYPE == 0
    cl_kernel kernelTunnel;
    cl_event kernelTunnel_event;
#endif
#if FILTERTYPE == 1 || FILTERTYPE == 3
    cl_kernel kernelGauss;
    cl_event kernelGauss_event[3];
#endif
#if FILTERTYPE == 2 || FILTERTYPE == 3
    cl_kernel kernelSobel;
    cl_event kernelSobel_event;
#endif

	int status;
	cl_mem input_buf, output_buf;
	
	platformToUse = findPlatform();

	device = obtainDevice(&platformToUse, 1, 0);

	context_properties[1] = (cl_context_properties)platformToUse;
	context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);
	char **opencl_program=read_file("videofilter.cl");
    program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
    if (program == NULL)
    {
            printf("Program creation failed\n");
            return 1;
    }
    int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
    
	// Input buffers
    input_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
       RESX*RESY*sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

	// Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        RESX*RESY*sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for output");

    // Kernel setup
#if FILTERTYPE == 0
    kernelTunnel = clCreateKernel(program, "tunnel", NULL);
    status = clSetKernelArg(kernelTunnel, 0, sizeof(cl_mem), &input_buf);
    checkError(status, "Failed to set argument 0");
    status = clSetKernelArg(kernelTunnel, 1, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 1");
#endif
#if FILTERTYPE == 1 || FILTERTYPE == 3
    kernelGauss = clCreateKernel(program, "gauss", NULL);
    status = clSetKernelArg(kernelGauss, 0, sizeof(cl_mem), &input_buf);
    checkError(status, "Failed to set argument 0");
    status = clSetKernelArg(kernelGauss, 1, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 1");
#endif
#if FILTERTYPE == 2 || FILTERTYPE == 3
    kernelSobel = clCreateKernel(program, "sobel", NULL);
    status = clSetKernelArg(kernelSobel, 0, sizeof(cl_mem), &input_buf);
    checkError(status, "Failed to set argument 0");
    status = clSetKernelArg(kernelSobel, 1, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 1");
#endif

	// Video setup
    VideoCapture camera("./bourne.mp4");
    if(!camera.isOpened())  // check if we succeeded
        return -1;

    const string NAME = "./output.avi";   // Form the new name with container
    int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
    Size S = Size((int) camera.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT));
	cout << "SIZE:" << S << endl;
	
    VideoWriter outputVideo;                                        // Open the output
        outputVideo.open(NAME, ex, 25, S, true);

    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << NAME << endl;
        return -1;
    }
	double diff;
    double tot = 0;
    struct timespec start_time, end_time;

    Mat cameraFrame, displayFrame;

	// Framewise computation
    for (int count=0; count < N_FRAMES; count ++) {
        // Obtain frame
        camera >> cameraFrame;

        clock_gettime(0, &start_time);
        Mat grayframe;
    	cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);

		//TODO border
		//copyMakeBorder(grayframe, grayframe, 1, 1, 1, 1, BORDER_REPLICATE);

		// Mat to array
        inputFrame = (unsigned char *)grayframe.data;
        
        // Write data to memory
        status = clEnqueueWriteBuffer(queue, input_buf, CL_TRUE,
            0, RESX*RESY*sizeof(char), inputFrame, 0, NULL, NULL);
        checkError(status, "Failed to transfer input");

        // Kernel
        const size_t global_work_size[2]= {RESX, RESY};
        const size_t local_work_size[2]= {16, 15};

    #if FILTERTYPE == 0
        status = clEnqueueNDRangeKernel(queue, kernelTunnel, 2, NULL,
        global_work_size, local_work_size, 0, NULL, &kernelTunnel_event);
        clWaitForEvents(1, &kernelTunnel_event);
    #endif
    #if FILTERTYPE == 1 || FILTERTYPE == 3
        // Gauss 1
        status = clEnqueueNDRangeKernel(queue, kernelGauss, 2, NULL,
            global_work_size, local_work_size, 0, NULL, &kernelGauss_event[0]);
        clWaitForEvents(1, &kernelGauss_event[0]);
        // swap in- and output
        status = clSetKernelArg(kernelGauss, 0, sizeof(cl_mem), &output_buf);
        status = clSetKernelArg(kernelGauss, 1, sizeof(cl_mem), &input_buf);
        // Gauss 2
        status = clEnqueueNDRangeKernel(queue, kernelGauss, 2, NULL,
            global_work_size, local_work_size, 0, NULL, &kernelGauss_event[0]);
        clWaitForEvents(1, &kernelGauss_event[1]);
        // swap in- and output
        status = clSetKernelArg(kernelGauss, 0, sizeof(cl_mem), &input_buf);
        status = clSetKernelArg(kernelGauss, 1, sizeof(cl_mem), &output_buf);
        // Gauss 3
        status = clEnqueueNDRangeKernel(queue, kernelGauss, 2, NULL,
            global_work_size, local_work_size, 0, NULL, &kernelGauss_event[2]);
        clWaitForEvents(1, &kernelGauss_event[2]);
    #endif
    #if FILTERTYPE == 2 || FILTERTYPE == 3
        // Sobel
        status = clEnqueueNDRangeKernel(queue, kernelSobel, 2, NULL,
        global_work_size, local_work_size, 0, NULL, &kernelSobel_event);
        clWaitForEvents(1, &kernelSobel_event);
    #endif  
        // Read output
        status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
            0, RESX*RESY*sizeof(char), filteredFrame, 0, NULL, NULL);
        checkError(status, "Failed to transfer output");
    	
		// array to Mat
        displayFrame = Mat(grayframe.size().height, grayframe.size().width, CV_8U, filteredFrame);
        cvtColor(displayFrame, displayFrame, CV_GRAY2BGR);

        clock_gettime(0, &end_time);
		outputVideo << displayFrame;

        diff =  1.0*(end_time.tv_sec - start_time.tv_sec)*1000 + 1.0/1000000 * (end_time.tv_nsec - start_time.tv_nsec);
		tot+=diff;
	}
	outputVideo.release();
	camera.release();
    printf("Runtime = %.2lfms\n", tot);
  	printf ("FPS %.2lf .\n", 1000*(N_FRAMES-1)/tot);

    //Cleanup
#if FILTERTYPE == 0
    clReleaseKernel(kernelTunnel);
#endif
#if FILTERTYPE == 1 || FILTERTYPE == 3
     clReleaseKernel(kernelGauss);
#endif
#if FILTERTYPE == 2 || FILTERTYPE == 3
     clReleaseKernel(kernelSobel);
#endif

    clReleaseCommandQueue(queue);
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseProgram(program);
    clReleaseContext(context);
    clFinish(queue);

	free(filteredFrame);

    return EXIT_SUCCESS;

}
