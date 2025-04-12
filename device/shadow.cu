#include <cuda_runtime.h>
#include <optix.h>

#include "main.h"

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
{}