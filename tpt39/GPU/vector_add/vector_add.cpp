#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#define STRING_BUFFER_LEN 1024
using namespace std;

// 0 ... Copy buffers
// 1 ... map buffers
#define MAP 1

void print_clbuild_errors(cl_program program,cl_device_id device) {
    cout<<"Program Build failed\n";
    size_t length;
    char buffer[2048];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
    cout<<"--- Build log ---\n "<<buffer<<endl;
    exit(1);
}

unsigned char ** read_file(const char *name) {
    size_t size;
    unsigned char **output = (unsigned char **)malloc(sizeof(unsigned char *));
    FILE* fp = fopen(name, "rb");
    if (!fp) {
        printf("no such file:%s",name);
        exit(-1);
    }

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    *output = (unsigned char *)malloc(size);
    unsigned char **outputstr=(unsigned char **)malloc(sizeof(unsigned char *));
    *outputstr= (unsigned char *)malloc(size);
    if (!*output) {
        fclose(fp);
        printf("mem allocate failure:%s",name);
        exit(-1);
    }

    if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
    fclose(fp);

    //printf("file size %d\n",size);
    //printf("-------------------------------------------\n");
    snprintf((char *)*outputstr,size,"%s\n",*output);
    //printf("%s\n",*outputstr);
    //printf("-------------------------------------------\n");

    return outputstr;
}

//Callback for GPU output
void callback(const char *buffer, size_t length, size_t final, void *user_data) {
     fwrite(buffer, 1, length, stdout);
}

void checkError(int status, const char *msg) {
    if(status!=CL_SUCCESS) {
        printf("%s\n",msg);
    }
}

// Randomly generate a floating-point number between -10 and 10, used for test data
float rand_float() {
    return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

int main() {
    char char_buffer[STRING_BUFFER_LEN];
    cl_platform_id platform;
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
    cl_kernel kernel;

    //--------------------------------------------------------------------
    //dimension of the vectors
    const unsigned N = 1E7;

    cl_mem input_a_buf; // num_devices elements
    cl_mem input_b_buf; // num_devices elements
    cl_mem output_buf; // num_devices elements
    cl_int errorcode = 0;
    int status;

# if MAP == 0
    // allocate memory
    float * input_a = (float * )malloc(sizeof(float)*N);
    float * input_b = (float * )malloc(sizeof(float)*N);
    float * output = (float * )malloc(sizeof(float)*N);
# endif

    //events for synchronization
    cl_event write_event[2];
    cl_event kernel_event;

    //variables for time measurement
    struct timespec start_time, end_time;
    double diff, diffWrite, diffRead,diffCPU;
    cl_ulong start_nanos, end_nanos, diff_nanos;

    //setup platform
    clGetPlatformIDs(1, &platform, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

    //setup context and queue
    context_properties[1] = (cl_context_properties)platform;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    //program and kernel creation
    unsigned char **opencl_program=read_file("vector_add.cl");
    program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
     if (program == NULL)
      {
             printf("Program creation failed\n");
             return 1;
      }
     int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
     if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
     kernel = clCreateKernel(program, "vector_add", NULL);

    // Input buffers
    input_a_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
       N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");

    // Set kernel arguments.
    unsigned argi = 0;
    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1");
    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2");
    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3");

# if MAP == 0
    //initialize inputs randomly
    for(unsigned i = 0; i < N; ++i) {
        input_a[i] = rand_float();
        input_b[i] = rand_float();
    }
# endif

    clock_gettime(0, &start_time);
# if MAP == 0
    // copy input
    status = clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE,
        0, N* sizeof(float), input_a, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue, input_b_buf, CL_FALSE,
        0, N* sizeof(float), input_b, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");
# endif
#if MAP == 1
    //Map the inputs
    float * input_a = (float *)clEnqueueMapBuffer(queue, input_a_buf, CL_TRUE,
        CL_MAP_WRITE,0, N* sizeof(float), 0, NULL, &write_event[0], &errorcode);
    checkError(errorcode, "Failed to map input A");
    float * input_b = (float *)clEnqueueMapBuffer(queue, input_b_buf, CL_TRUE,
        CL_MAP_WRITE, 0,N* sizeof(float), 0, NULL, &write_event[1], &errorcode);
    checkError(errorcode, "Failed to map input B");
#endif    
    clWaitForEvents(2, write_event);
    clock_gettime(0, &end_time);

    diffWrite = 1.0*(end_time.tv_sec - start_time.tv_sec)*1000 + 1.0/1000000 * (end_time.tv_nsec - start_time.tv_nsec);
    checkError(errorcode, "Failed to transfer input A");
    checkError(errorcode, "Failed to transfer input B");

# if MAP == 1
    //initialize inputs randomly
    for(unsigned i = 0; i < N; ++i) {
        input_a[i] = rand_float();
        input_b[i] = rand_float();
    }
# endif

     //reference calcualtion (CPU)
    float *ref_output=(float *) malloc(sizeof(float)*N);
    clock_gettime(0, &start_time);
    for(unsigned j = 0; j < N; ++j) {
        ref_output[j] = input_a[j] + input_b[j];
    }
    clock_gettime(0, &end_time);
    diffCPU = 1.0*(end_time.tv_sec - start_time.tv_sec)*1000 + 1.0/1000000 * (end_time.tv_nsec - start_time.tv_nsec);

# if MAP == 1
    // unmap
    clEnqueueUnmapMemObject(queue,input_a_buf,input_a,0,NULL, NULL);
    clEnqueueUnmapMemObject(queue,input_b_buf,input_b,0,NULL, NULL);
# endif

    // GPU calculation
    clock_gettime(0, &start_time);
    const size_t global_work_size = N;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
        &global_work_size, NULL, 2, write_event, &kernel_event);
    clWaitForEvents(1, &kernel_event);
    clock_gettime(0, &end_time);
    checkError(status, "Failed to launch kernel");
    diff = 1.0*(end_time.tv_sec - start_time.tv_sec)*1000 + 1.0/1000000 * (end_time.tv_nsec - start_time.tv_nsec);

    clock_gettime(0, &start_time);
# if MAP == 0
    // copy output
    status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
        0, N* sizeof(float), output, 1, &kernel_event, NULL);
# endif
# if MAP == 1
    // map output
    clock_gettime(0, &start_time);
    float * output = (float *)clEnqueueMapBuffer(queue, output_buf, CL_TRUE,
        CL_MAP_READ, 0,N* sizeof(float),  0, NULL, NULL, &errorcode);
    checkError(errorcode, "Failed to write output");
# endif
    clock_gettime(0, &end_time);
    diffRead = 1.0*(end_time.tv_sec - start_time.tv_sec)*1000 + 1.0/1000000 * (end_time.tv_nsec - start_time.tv_nsec);

    // Calculate runtime kernel with built-in method
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start_nanos), &start_nanos, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end_nanos), &end_nanos, NULL);
    diff_nanos = end_nanos-start_nanos;

    // Print runtimes
    printf("With N = %u\n", N);
    printf ("CPU took %.2lfms to run.\n", diffCPU);
    printf ("GPU took %.2lfms (Write data).\n", diffWrite);
    printf ("GPU took %.2lfms (Kernel).\n", diff);
    printf ("GPU took %.2lfms (Read data).\n", diffRead);
    printf ("Overall GPU time is %.2lfms.\n", diffRead + diffWrite + diff);
    printf("GPU Kernel computation time (kernel only): %.2lfms\n", 1.0*diff_nanos/1000000);

    // Verify results.
    bool pass = true;
    for(unsigned j = 0; j < N && pass; ++j) {
         if(fabsf(output[j] - ref_output[j]) > 1.0e-5f) {
           printf("Failed verification @ index %d\nOutput: %f\nReference: %f\n",
               j, output[j], ref_output[j]);
           pass = false;
         }
    }

# if MAP == 0
    free(input_a);
    free(input_b);
    free(output);
# endif
# if MAP ==1
    clEnqueueUnmapMemObject(queue,output_buf,output,0,NULL,NULL);
# endif

    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(input_a_buf);
    clReleaseMemObject(input_b_buf);
    clReleaseMemObject(output_buf);
    clReleaseProgram(program);
    clReleaseContext(context);
    clFinish(queue);

    return 0;
}
