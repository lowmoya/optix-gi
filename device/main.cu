#include <cuda_runtime.h>
#include <optix.h>

#include "main.h"

extern "C"
{
__constant__ Params params;
}

extern "C"
__global__ void __raygen__gi()
{
    /* Generate the ray. */
    const uint3 launch_index = optixGetLaunchIndex();
    const float3 ray_origin = params.cam_eye;
    const float3 ray_direction = normalized(
        params.cam_w
        + (float)(2.0f * ((float)launch_index.x + 0.5)
            / (float)params.image_width - 1.0f) * params.cam_u
        + (float)(2.0f * ((float)launch_index.y + 0.5)
            / (float)params.image_height - 1.0f) * params.cam_v
    );
 
    /* Sample scene. */
    unsigned int p0, p1, p2;
    optixTrace(params.handle, ray_origin, ray_direction,
        0.0f, 1e16f, 0.0, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        0, 0, 0, p0, p1, p2);
        

    /* Write result */
    float3 result = {
        __int_as_float(p0),
        __int_as_float(p1),
        __int_as_float(p2)
    };
    uchar4 result_color = {
        (u_char)(result.x * 255),
        (u_char)(result.y * 255),
        (u_char)(result.z * 255),
        255
    };
    params.image[launch_index.y * params.image_width + launch_index.x] =
        result_color;
}

extern "C"
__global__ void __closesthit__gi()
{
    const float2 barycentrics = optixGetTriangleBarycentrics();
    optixSetPayload_0(__float_as_int(barycentrics.x));
    optixSetPayload_1(__float_as_int(barycentrics.y));
    optixSetPayload_2(__float_as_int(1.0f));
}

extern "C"
__global__ void __miss__gi()
{
    optixSetPayload_0(__float_as_int(0.0f));
    optixSetPayload_1(__float_as_int(0.0f));
    optixSetPayload_2(__float_as_int(0.0f)); 
}