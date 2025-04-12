#include <cuda_runtime.h>
#include <optix.h>

#include "main.h"

extern "C"
{
__constant__ Params params;
}

extern "C"
__global__ void __raygen__radiance()
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
        RT_RADIANCE, RT_COUNT, MT_RADIANCE, p0, p1, p2);
        

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

// Rewrite at someponit
__device__
float3 calcColor(float3 incoming, float3 outgoing, float3 normal, float roughness, float3 F0, float3 color)
{
    float3 h = normalized(incoming + outgoing);
    float NdotL = max(dot(normal, outgoing), 0.0);
    float NdotV = max(dot(normal, incoming), 0.0);
    float NdotH = max(dot(normal, h), 0.0);
    float VdotH = max(dot(incoming, h), 0.0);

    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (alpha2 - 1.0) + 1.0;
    float D = alpha2 / (M_PIf * denom * denom);

    float3 F = F0 + (make_float3(1.0, 1.0, 1.0) - F0) * pow(1.0 - VdotH, 5);

    float G_V = NdotV / (NdotV + sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV));
    float G_L = NdotL / (NdotL + sqrt(alpha2 + (1.0f - alpha2) * NdotL * NdotL));
    float G = G_V * G_L;

    float3 specular = (D * G * F) / (4.0f * NdotL * NdotV + 1e-5f);

    float3 kd = make_float3(1, 1, 1) - F;
    float3 diffuse = kd * color / M_PIf;


    return (diffuse + specular) * make_float3(.8, .8, .8) * NdotL;
}

extern "C"
__global__ void __closesthit__radiance()
{
    const float2 bc = optixGetTriangleBarycentrics();
    const HitGroupData & group_data = *reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
    
    const uint3 indices = group_data.indices[optixGetPrimitiveIndex()];
    const float3 position = optixHitObjectTransformPointFromObjectToWorldSpace((1 - bc.x - bc.y) * group_data.vertices[indices.x]
        + bc.x * group_data.vertices[indices.y]
        + bc.y * group_data.vertices[indices.z]);
    const float3 normal = normalized(optixHitObjectTransformNormalFromWorldToObjectSpace((1 - bc.x - bc.y) * group_data.normals[indices.x]
        + bc.x * group_data.normals[indices.y]
        + bc.y * group_data.normals[indices.z]));

    const float3 light = make_float3(0, 6, 0);

    const float3 incoming = optixGetWorldRayDirection();
    const float3 outgoing = normalized(light - position);

    uint unobstructed = 0;
    optixTrace(params.handle, position, outgoing, 0.0001, magnitude(light - position), 0, OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, RT_SHADOW, RT_COUNT, MT_SHADOW, unobstructed);
    const float3 color = unobstructed ? calcColor(incoming, outgoing, normal, .2, make_float3(0.04, 0.04, 0.04), group_data.color)
        * 500 / pow(magnitude(light), 2) * dot(normal, outgoing) : make_float3(0, 0, 0);
    // const float3 color = unobstructed ? make_float3(1, 1, 1) : make_float3(0, 0, 0);
    optixSetPayload_0(__float_as_int(min(color.x, 1.0)));
    optixSetPayload_1(__float_as_int(min(color.y, 1.0)));
    optixSetPayload_2(__float_as_int(min(color.z, 1.0)));
}

extern "C"
__global__ void __miss__radiance()
{
    optixSetPayload_0(__float_as_int(0.0f));
    optixSetPayload_1(__float_as_int(0.0f));
    optixSetPayload_2(__float_as_int(0.0f)); 
}