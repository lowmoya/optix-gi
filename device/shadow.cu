#include <cuda_runtime.h>
#include <optix.h>

#include "common.h"

// Simple obstruction module

extern "C"
{
__constant__ Params params;
}

extern "C"
__global__ void __anyhit__shadow()
{
    optixSetPayload_0(0);
    optixTerminateRay();
}

extern "C"
__global__ void __miss__shadow()
{
    optixSetPayload_0(1);
}