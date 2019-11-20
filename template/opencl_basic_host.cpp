#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// TODO: do i need this?
#include <CL/cl_ext.h>


// structure following Jülich guideline

// TODO introduce some error handling
cl_int findPlatforms(cl_platform_id* platforms, cl_uint* platformCountPointer) {
    cl_uint i, j;
    char* info;
    size_t infoSize;

    cl_uint platformCount = *platformCountPointer;

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
    return 0;
}

cl_device_id obtainFirstDevice(cl_platform_id* platforms, cl_uint* platformCountPointer, int deviceIndex) {
    cl_uint i, j;
    char* value;
    size_t valueSize;
    cl_uint maxComputeUnits;

    cl_uint deviceCount;
    cl_device_id* devices = NULL;
    cl_device_id deviceToUse;

    cl_uint platformCount = *platformCountPointer;

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

int main() {

    // 1 Determine components
    // Code from https://gist.github.com/courtneyfaulkner/7919509
    
        cl_uint platformCount;
        cl_platform_id* platforms;
        cl_device_id deviceToUse;

        // 1.1 Query platforms

        platforms = NULL;
        findPlatforms(platforms, &platformCount);

        // 1.2 Query devices

        deviceToUse = obtainFirstDevice(platforms, &platformCount, 0);

    // 2 Query specific component properties, adapt program accordingly

        // 2.1 Create context for the devices

        // 2.2 Create queue

        // 2.3 Create program object

    // 3 Create and initialize memory objects


    // 4 Configurem compile and execute kernel(s)

        // 4.1 Set arguments

        // 4.2 Put Kernel into queue

 

    // 6 Collect results

        //4.1 Copy/Map memory objects from device to host

    // 7 Clean up

        free(platforms);
        return 0;

}