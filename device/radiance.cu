#include <cuda_runtime.h>
#include <curand_kernel.h>
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
    unsigned int p0, p1, p2, depth = 0;
    optixTrace(params.handle, ray_origin, ray_direction,
        0.0f, 1e16f, 0.0, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        RT_RADIANCE, RT_COUNT, MT_RADIANCE, p0, p1, p2, depth);
        

    /* Write result */
    float3 result = {
        __int_as_float(p0),
        __int_as_float(p1),
        __int_as_float(p2)
    };
    uchar4 result_color = {
        (u_char)(powf(result.x, 1 / 2.2) * 255),
        (u_char)(powf(result.y, 1 / 2.2) * 255),
        (u_char)(powf(result.z, 1 / 2.2) * 255),
        255
    };
    params.image[launch_index.y * params.image_width + launch_index.x] =
        result_color;
}

// Rewrite at someponit
__device__
float3 brdf(float3 incoming, float3 outgoing, float3 normal, float roughness, float metallic, float3 color)
{
    float3 h = normalized(incoming + outgoing);
    float NdotL = max(dot(normal, outgoing), 0.0);
    float NdotV = max(dot(normal, incoming), 0.0);
    float NdotH = max(dot(normal, h), 0.0);
    float VdotH = max(dot(incoming, h), 0.0);

    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (alpha2 - 1.0) + 1.0;
    float D = alpha2 / (M_PIf * denom * denom + 1e-5);

    float3 F0 = make_float3(
            .04 + metallic * (color.x - .04),
            .04 + metallic * (color.y - .04),
            .04 + metallic * (color.z - .04)
    );
    float3 F = F0 + (make_float3(1.0, 1.0, 1.0) - F0) * powf(1.0 - VdotH, 5.0);

    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float G_V = NdotV / (NdotV  * (1.0f - k)  + k + 1e-5);
    float G_L = NdotL / (NdotL  * (1.0f - k)  + k + 1e-5);
    float G = G_V * G_L;

    float3 specular = (D * G * F) / (4.0f * NdotL * NdotV + 1e-5f);

    float3 kd = (make_float3(1, 1, 1) - F) * (1 - metallic);
    float3 diffuse = kd * color / M_PIf;


    return diffuse + specular;
    // return color / M_PI;
}

// Rewrite
// change sampling method to require less samples
__device__ float3 sampleHemisphereUniform(float3 normal, float u1, float u2) {
    float r = sqrtf(u1);
    float theta = 2 * M_PIf * u2;

    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(fmaxf(0, 1 - u1));

    float3 normal_u, normal_v;
    if (fabs(normal.z) > 0.999f) {
        normal_u = make_float3(1, 0, 0);
        normal_v = make_float3(0, 1, 0);
    } else {
        normal_u = normalized(cross(make_float3(0, 0, 1), normal));
        normal_v = cross(normal, normal_u);
    }

    return normalized(normal_u * x + normal_v * y + normal * z);
}
__constant__ __uint32_t sobol_matrix[4][32] = {
    // Dimension 0
    {
        0x80000000, 0x40000000, 0x20000000, 0x10000000,
        0x08000000, 0x04000000, 0x02000000, 0x01000000,
        0x00800000, 0x00400000, 0x00200000, 0x00100000,
        0x00080000, 0x00040000, 0x00020000, 0x00010000,
        0x00008000, 0x00004000, 0x00002000, 0x00001000,
        0x00000800, 0x00000400, 0x00000200, 0x00000100,
        0x00000080, 0x00000040, 0x00000020, 0x00000010,
        0x00000008, 0x00000004, 0x00000002, 0x00000001
    },
    // Dimension 1
    {
        0x80000000, 0xC0000000, 0x60000000, 0x30000000,
        0x18000000, 0x0C000000, 0x06000000, 0x03000000,
        0x01800000, 0x00C00000, 0x00600000, 0x00300000,
        0x00180000, 0x000C0000, 0x00060000, 0x00030000,
        0x00018000, 0x0000C000, 0x00006000, 0x00003000,
        0x00001800, 0x00000C00, 0x00000600, 0x00000300,
        0x00000180, 0x000000C0, 0x00000060, 0x00000030,
        0x00000018, 0x0000000C, 0x00000006, 0x00000003
    },
};
__device__ __uint32_t sobol(__uint32_t index, int dim) {
    __uint32_t result = 0;
    for (int i = 0; i < 32; ++i) {
        if (index & (1u << i))
            result ^= sobol_matrix[dim][i];
    }
    return result;
}

extern "C"
__global__ void __closesthit__radiance()
{
    const float2 bc = optixGetTriangleBarycentrics();
    const HitGroupData & group_data = *reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
    const uint depth = optixGetPayload_3();
    
    
    const uint3 indices = group_data.indices[optixGetPrimitiveIndex()];
    const float3 position = optixHitObjectTransformPointFromObjectToWorldSpace((1 - bc.x - bc.y) * group_data.vertices[indices.x]
        + bc.x * group_data.vertices[indices.y]
        + bc.y * group_data.vertices[indices.z]);
    const float3 normal = normalized(optixHitObjectTransformNormalFromWorldToObjectSpace((1 - bc.x - bc.y) * group_data.normals[indices.x]
        + bc.x * group_data.normals[indices.y]
        + bc.y * group_data.normals[indices.z]));

    /* Direct lighting */
    struct {
        float3 position;
        float3 width_dir;
        float3 height_dir;
        float width;
        float height;
        float3 color;
        float intensity;
        int samples;
    } lights[] = {
        {
            make_float3(-10, 10, 0),
            make_float3(1, 0, 0), normalized(make_float3(.3, 1, 0)),
            10, 10,
            make_float3(1.0, .64, .4),
            900,
            // 200
            2
        },
        {
            make_float3(0,6.9,0),
            make_float3(1, 0, 0), make_float3(0, 0, 1),
            .3, .3,
            make_float3(1.0, 1.0, 1.0),
            40,
            // 10
            1
        }
    };
    const int light_count = sizeof(lights)/sizeof(*lights);

    float3 direct_color = make_float3(0, 0, 0);
    const float3 outgoing = -1 * optixGetWorldRayDirection();
    for (int i = 0; i < light_count; ++i) {
        for (int s = 0; s < lights[i].samples; ++s) {
            const float3 sample = lights[i].position
                + ((double)sobol(s, 0) / (double)__UINT32_MAX__) * lights[i].width_dir * lights[i].width
                + ((double)sobol(s, 1) / (double)__UINT32_MAX__) * lights[i].height_dir * lights[i].height;
            const float3 incoming = sample - position;
            const float incoming_mag = magnitude(incoming);
            const float3 incoming_dir = incoming / incoming_mag;

            uint unobstructed = 0;
            optixTrace(params.handle, position, incoming_dir, 0.001, incoming_mag, 0, OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, RT_SHADOW, RT_COUNT, MT_SHADOW, unobstructed);

            if (!unobstructed)
                continue;
            
            float3 brdf_val = brdf(incoming_dir, outgoing, normal, group_data.roughness, group_data.metallic,
                group_data.color);
            direct_color += lights[i].intensity * lights[i].color * brdf_val * dot(normal, incoming_dir)
                / (incoming_mag * incoming_mag) / (float)lights[i].samples;
        }
    }

    
    curandState rand_state;
    /* Indirect lighting */
    float3 indirect_color = make_float3(0,0,0);
    if (depth < MAX_TRACING_DEPTH - 1) {
        // const int samples = 60;
        const int samples = 10;
        for (int s = 0; s < samples; ++s) { 
            uint next_depth = depth + 1;
            uint3 sample_color;
            // float3 sample_direction = -1 * sampleHemisphereUniform(normal, curand_uniform(&rand_state),
            //     curand_uniform(&rand_state));
            float3 sample_direction = sampleHemisphereUniform(normal,
                (double)sobol(s, 0) / (double)__UINT32_MAX__, (double)sobol(s, 1) / (double)__UINT32_MAX__);

            optixTrace(params.handle, position, sample_direction, 0.001, 1e16, 0, OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE, RT_RADIANCE, RT_COUNT, MT_RADIANCE, sample_color.x,
                sample_color.y, sample_color.z, next_depth);

            indirect_color = indirect_color + (1.0 / (float)samples)
                * brdf(sample_direction, outgoing, normal, group_data.roughness,
                    group_data.metallic, group_data.color)
                * make_float3(__int_as_float(sample_color.x), __int_as_float(sample_color.y),
                    __int_as_float(sample_color.z))
                * make_float3(1, 1, 1)
                * dot(sample_direction,normal);
        }
    }


    /* Return color*/
    const float3 color = (direct_color + indirect_color);
    // scale color by the world rays distance
    optixSetPayload_0(__float_as_int(min(color.x, 1.0)));
    optixSetPayload_1(__float_as_int(min(color.y, 1.0)));
    optixSetPayload_2(__float_as_int(min(color.z, 1.0)));
}

extern "C"
__global__ void __miss__radiance()
{
    float t = (optixGetWorldRayDirection().y + 1.0f) / 2.0;

    // Simple 2-color gradient: horizon to zenith
    float3 zenith = make_float3(0.2f, 0.2f, 0.4f);
    float3 horizon = make_float3(1.0f, 0.5f, 0.2f);
    float3 sky_color = zenith + t * (horizon - zenith);

    optixSetPayload_0(__float_as_int(1.f));
    optixSetPayload_1(__float_as_int(0.5f));
    optixSetPayload_2(__float_as_int(.2f)); 
}