#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>


// structure following Jülich guideline
/*int main() {
    // 1 Determine components
    cl_uint num_platforms = 0;
    cl_platform_id* platforms;

    // first call to obtain number of platforms
    clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS;
    
    platforms = (cl_platform_id *) malloc(num_platforms * sizeof(cl_platfor_id));

    // write platforms
    clGetPlatformIDs(num_platforms, platforms, NULL) != CL_SUCCESS);

    // 2 Query specific component properties, adapt program accordingly

    // 3 Configure and compile kernel(s)

    // 4 Create and initialize memory objects

    // 5 Execute kernel

    // 6 Collect results
}*/
int main() {

    int i, j;
    char* info;
    size_t infoSize;
    cl_uint platformCount;
    cl_platform_id *platforms;
    const char* attributeNames[5] = { "Name", "Vendor",
        "Version", "Profile", "Extensions" };
    const cl_platform_info attributeTypes[5] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
        CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };
    const int attributeCount = sizeof(attributeNames) / sizeof(char*);

    // get platform count
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

    free(platforms);
    return 0;

}