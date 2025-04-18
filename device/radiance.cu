#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <optix.h>

#include "common.h"
#include "extmath.h"

extern "C"
{
__constant__ Params params;
}

/*
 * Utility functinos
 */
__device__ void storePointer(void * ptr, uint32_t & lh, uint32_t & rh) {
    uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    lh = static_cast<uint32_t>(uptr >> 32);
    rh = static_cast<uint32_t>(uptr);
}
__device__ void * loadPointer(uint32_t lh, uint32_t rh) {
    uint64_t uptr = (static_cast<uint64_t>(lh) << 32)
        | (static_cast<uint64_t>(rh));
    return reinterpret_cast<void *>(uptr);
}
__device__
float minf(float a, float b) {
    return a > b ? b : a;
}
__device__
float maxf(float a, float b) {
    return a > b ? a : b;
}
// Based on distributions from
// https://en.wikipedia.org/wiki/Specular_highlight
// https://en.wikipedia.org/wiki/Lambertian_reflectance
__device__
float spectrumBRDF(float3 incoming, float3 outgoing, float3 normal, float albedo, float roughness, float metallic)
{
    // Calculate common variables
    float3 h = normalized(incoming + outgoing);
    float dLN = max(dot(normal, outgoing), 0.0001);
    float dVN = max(dot(normal, incoming), 0.0001);
    float dHN = min(max(dot(normal, h), 0.0001), 1.0);
    float dVH = max(dot(incoming, h), 0.0001);
    roughness = max(roughness, 0.0001);

    // Calculate fresnel reflectance
    float F0 = .04 + metallic * (albedo - .04);
    float F = F0 + (1 - F0) * powf(1.0 - dVH, 5.0);


    // Beckmann distribution
    float alpha = acosf(dHN);
    float cos2a = powf(cosf(alpha), 2);
    float tan2a = (1.0 - cos2a) / cos2a;
    float beckmann = expf(-tan2a / (roughness * roughness)) / (M_PIf * roughness * roughness * cos2a * cos2a);


    // Cook-Torrance model
    float G = minf(minf(
        (2.0 * dHN * dVN) / dVH,
        (2.0 * dHN * dLN) / dVH
    ), 1);
    float kspec = (beckmann * G * F) / (M_PIf * dVN * dLN); 

    // Diffuse model
    float kd = (1 - F) * (1 - metallic) * (float)albedo / M_PIf;

    return kspec + kd;
}

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
/* Estimate a spectrum from some RGB*/
__device__ float guassian(float wavelength, float center, float distribution) {
    float difference = (wavelength - center) / distribution;
    return expf(-0.5 * difference * difference);
}
__device__ void makeSpectrum(float3 color, float * samples) {
    for (int i = 0; i < SPECTRAL_SAMPLES; ++i) {
        const int wavelength = SPECTRAL_START + i * SPECTRAL_STEP;
        samples[i] += (color.z * guassian(wavelength, 450, 30)
            + color.y * guassian(wavelength, 530, 30)
            + color.x * guassian(wavelength, 620, 30));
    }
}


/*
 * RAY GEN
 */
extern "C"
__global__ void __raygen__radiance()
{
    /* Generate the ray. */
    const uint3 launch_index = optixGetLaunchIndex();
    const float3 ray_origin = params.cam_pos;
    const float3 ray_direction = normalized(
        params.cam_w
        + (float)(2.0f * ((float)(launch_index.x + params.offset_x) + 0.5)
            / (float)params.output_width - 1.0f) * params.cam_u
        + (float)(2.0f * ((float)(launch_index.y + params.offset_y) + 0.5)
            / (float)params.output_height - 1.0f) * params.cam_v
    );
    unsigned int spectra_index =
        ((launch_index.x + params.offset_x)
        + params.output_width * (launch_index.y + params.offset_y))
            * SPECTRAL_SAMPLES;

    /* Initialize spectra to 0. */
    for (int i = 0; i < SPECTRAL_SAMPLES; ++i)
        params.spectra[spectra_index + i] = 0;

    /* Prepare payload. */
    unsigned int ptr_lh, ptr_rh;
    storePointer(params.spectra + spectra_index, ptr_lh, ptr_rh);

    unsigned int depth = 1;


    optixTrace(params.handle, ray_origin, ray_direction, 0.0f, 1e16f,
        0.0, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        RT_RADIANCE, RT_COUNT, MT_RADIANCE,
        depth, ptr_lh, ptr_rh);

}


/*
 * CLOSEST HIT
 */
struct Light {
    float3 position;
    float3 width_dir;
    float3 height_dir;
    float width;
    float height;
    float intensity;
    int samples;
    int sun;
    float spectrum[SPECTRAL_SAMPLES];
};
extern "C"
__global__ void __closesthit__radiance()
{
    // Get intersection data
    const float2 bc = optixGetTriangleBarycentrics();
    const HitGroupData & group_data = *reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
    
    // Get payload params
    const uint depth = optixGetPayload_0();
    float * spectrum = reinterpret_cast<float *>(loadPointer(optixGetPayload_1(), optixGetPayload_2()));


    // Extract geometry data
    const uint3 indices = group_data.indices[optixGetPrimitiveIndex()];
    const float3 position = optixTransformPointFromObjectToWorldSpace((1 - bc.x - bc.y) * group_data.vertices[indices.x]
        + bc.x * group_data.vertices[indices.y]
        + bc.y * group_data.vertices[indices.z]);
    const float3 normal = normalized(optixTransformNormalFromObjectToWorldSpace((1 - bc.x - bc.y) * group_data.normals[indices.x]
        + bc.x * group_data.normals[indices.y]
        + bc.y * group_data.normals[indices.z]));
    const float2 uv = (1 - bc.x - bc.y) * group_data.uv[indices.x]
        + bc.x * group_data.uv[indices.y]
        + bc.y * group_data.uv[indices.z]; 

    const float4 texture = tex2D<float4>(group_data.texture, uv.x, uv.y);
    const float3 albedo = make_float3(texture.x, texture.y, texture.z);
    float albedo_spectrum[SPECTRAL_SAMPLES] = {};
    makeSpectrum(albedo, albedo_spectrum);

    /* Direct lighting */
    Light lights[] = {
        {
            normalized(make_float3(-1, 2, 0)), make_float3(0, 0, 0), make_float3(0, 0, 0),
            10, 10,
            3, 1, 1,
            {
                0.02f, 0.05f, 0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f,
                0.70f, 0.80f, 0.90f, 1.00f, 0.90f, 0.80f, 0.70f, 0.60f 
            }
        },
        {
            make_float3(0,6.9,0),
            make_float3(1, 0, 0), make_float3(0, 0, 1),
            .3, .3,
            60, 1 * LIGHT_SAMPLES, 0,
            {
                0.05, 0.15, 0.25, 0.45, 0.75, 1.00, 1.00, 1.00,
                0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.30, 0.10
            }
        }
    };
    int light_count = sizeof(lights) / sizeof(Light);
    
    const float3 outgoing = -1 * optixGetWorldRayDirection();
    
    for (int li = 0; li < light_count; ++li) {
        for (int s = 0; s < lights[li].samples; ++s) {
            float incoming_mag;
            float3 incoming_dir;
            uint unobstructed = 0;
            if (!lights[li].sun) {
                // Select the incoming dir at random
                float3 sample = lights[li].position
                    + ((double)sobol(s, 0) / (double)__UINT32_MAX__) * lights[li].width_dir * lights[li].width
                    + ((double)sobol(s, 1) / (double)__UINT32_MAX__) * lights[li].height_dir * lights[li].height;
                float3 incoming = sample - position;
                incoming_mag = magnitude(incoming);
                incoming_dir = normalized(incoming / incoming_mag);
                optixTrace(params.handle, position, incoming_dir, 0.001, incoming_mag, 0, OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, RT_SHADOW, RT_COUNT, MT_SHADOW, unobstructed);
            } else {
                // Select the point as incoming dir
                // Remove distance from equation
                incoming_mag = 1;
                incoming_dir = lights[li].position;
                optixTrace(params.handle, position, incoming_dir, 0.001, 1e16, 0, OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, RT_SHADOW, RT_COUNT, MT_SHADOW, unobstructed);
            }
            if (!unobstructed)
                continue;
            
            // Apply contribution from lightsource hitting surface to total
            if (depth == 1) {
                float light_scale = lights[li].intensity * dot(normal, incoming_dir)
                    / (incoming_mag * incoming_mag * (float)lights[li].samples);
                
                for (int si = 0; si < SPECTRAL_SAMPLES; ++si) {
                    spectrum[si] += light_scale * lights[li].spectrum[si]
                        * spectrumBRDF(incoming_dir, outgoing, normal, albedo_spectrum[si], group_data.roughness, group_data.metallic);
                }
            } else {
                incoming_mag = max(incoming_mag, 2.0);
                for (int si = 0; si < SPECTRAL_SAMPLES; ++si) {
                    spectrum[si] += 
                        (lights[li].intensity * lights[li].spectrum[si] * dot(normal, incoming_dir)
                        * spectrumBRDF(incoming_dir, outgoing, normal, albedo_spectrum[si], group_data.roughness, group_data.metallic))
                        / (incoming_mag * incoming_mag * lights[li].samples);
                }
            }
        }
    }

    
    /* Indirect lighting */
    if (depth < MAX_TRACING_DEPTH - 1) {
        for (int s = 0; s < INDIRECT_SAMPLES; ++s) { 
            float3 sample_direction = sampleHemisphereUniform(normal,
                (double)sobol(s, 0) / (double)__UINT32_MAX__, (double)sobol(s, 1) / (double)__UINT32_MAX__);

            /* Sample a spectrum from a random direction*/
            float in_spectrum[SPECTRAL_SAMPLES] = {};
            uint next_depth = depth + 1;
            uint ptr_lh, ptr_rh;
            storePointer(in_spectrum, ptr_lh, ptr_rh);
            optixTrace(params.handle, position, sample_direction, 0.001, 1e16, 0, OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE, RT_RADIANCE, RT_COUNT, MT_RADIANCE, next_depth, ptr_lh, ptr_rh);

            /* Apply contribution from spectrum to total */
            for (int si = 0; si < SPECTRAL_SAMPLES; ++si) {
                spectrum[si] +=
                    in_spectrum[si] * dot(normal, sample_direction)
                    * spectrumBRDF(sample_direction, outgoing, normal, albedo_spectrum[si], group_data.roughness, group_data.metallic)
                    / (float)INDIRECT_SAMPLES;
            }
        }
    }
}


/*
 * MISS
 */

extern "C"
__global__ void __miss__radiance()
{
    // Get environment sample
    const MissData * data = (MissData *)optixGetSbtDataPointer();
    const float3 dir = normalized(optixGetWorldRayDirection());

    const float u = (atan2f(dir.x + 1.2 , dir.z) + M_PIf) * (0.5f * M_1_PIf);
    const float v = 0.5f * (1.0f + sin(M_PIf * 0.5f - acosf(dir.y)));

    float4 tex = tex2D<float4>(data->environment, u, v);
    float3 color = make_float3(tex.x, tex.y, tex.z);


    // Apply spectrum
    float * spectrum = reinterpret_cast<float *>(loadPointer(optixGetPayload_1(), optixGetPayload_2()));
    makeSpectrum(color, spectrum);
}