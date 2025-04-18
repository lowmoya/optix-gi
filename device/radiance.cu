#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <optix.h>

#include "main.h"

extern "C"
{
__constant__ Params params;
}

/*
 * RAY GEN
 */

extern "C"
__global__ void __raygen__radiance()
{
    /* Generate the ray. */
    const uint3 launch_index = optixGetLaunchIndex();
    const float3 ray_origin = params.cam_eye;
    const float3 ray_direction = normalized(
        params.cam_w
        + (float)(2.0f * ((float)launch_index.x + 0.5)
            / (float)params.output_width - 1.0f) * params.cam_u
        + (float)(2.0f * ((float)launch_index.y + 0.5)
            / (float)params.output_height - 1.0f) * params.cam_v
    ); 
 
    /* Sample scene. */
    unsigned int    spectra_index = (launch_index.x + params.output_width * launch_index.y) * SPECTRAL_SAMPLES,
                    cont_r = __float_as_uint(1.0),
                    cont_g = __float_as_uint(1.0),
                    cont_b = __float_as_uint(1.0),
                    depth = 1;

    /* Initialize spectra to 0. */
    for (int i = 0; i < SPECTRAL_SAMPLES; ++i)
        params.spectra[spectra_index + i] = 0.0;

    optixTrace(params.handle, ray_origin, ray_direction, 0.0f, 1e16f,
        0.0, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        RT_RADIANCE, RT_COUNT, MT_RADIANCE,
        spectra_index, cont_r, cont_g, cont_b, depth);
}


/*
 * CLOSEST HIT
 */

__device__
float minf(float a, float b) {
    return a > b ? b : a;
}

// Based on distributions from
// https://en.wikipedia.org/wiki/Specular_highlight
// https://en.wikipedia.org/wiki/Lambertian_reflectance
__device__
float3 albedoBRDF(float3 incoming, float3 outgoing, float3 normal, float3 albedo, float roughness, float metallic)
{
    // Calculate common variables
    float3 h = normalized(incoming + outgoing);
    float dLN = max(dot(normal, outgoing), 0.0);
    float dVN = max(dot(normal, incoming), 0.0);
    float dHN = max(dot(normal, h), 0.0);
    float dVH = max(dot(incoming, h), 0.0);

    // Calculate fresnel reflectance
    float3 F0 = make_float3(
        .04 + metallic * (albedo.x - .04),
        .04 + metallic * (albedo.y - .04),
        .04 + metallic * (albedo.z - .04)
    );
    float3 F = F0 + (make_float3(1.0, 1.0, 1.0) - F0) * powf(1.0 - dVH, 5.0);


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
    float3 kspec = (beckmann * G * F) / (M_PIf * dVN * dLN); 

    // Diffuse model
    float3 kd = (make_float3(1, 1, 1) - F) * (1 - metallic) * albedo / M_PIf;

    return kspec + kd;
}
__device__
float spectrumBRDF(float3 incoming, float3 outgoing, float3 normal, float albedo, float roughness, float metallic)
{
    // Calculate common variables
    float3 h = normalized(incoming + outgoing);
    float dLN = max(dot(normal, outgoing), 0.0);
    float dVN = max(dot(normal, incoming), 0.0);
    float dHN = max(dot(normal, h), 0.0);
    float dVH = max(dot(incoming, h), 0.0);

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

/* Estimate a spectrum from some RGB*/
__device__ float guassian(float wavelength, float center, float distribution) {
    float difference = (wavelength - center) / distribution;
    return expf(-0.5 * difference * difference);
}
__device__ void makeSpectrum(float3 color, float * spectrum) {
    for (int i = 0; i < SPECTRAL_SAMPLES; ++i) {
        const int wavelength = SPECTRAL_START + i * SPECTRAL_STEP;
        spectrum[i] += color.z * guassian(wavelength, 440, 30)
            + color.y * guassian(wavelength, 540, 30) + color.x * guassian(wavelength, 610, 30);
    }
}


extern "C"
__global__ void __closesthit__radiance()
{
    // Get intersection data
    const float2 bc = optixGetTriangleBarycentrics();
    const HitGroupData & group_data = *reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
    
    // Get payload params
    const uint spectra_index = optixGetPayload_0();
    const float cont_r = __uint_as_float(optixGetPayload_1());
    const float cont_g = __uint_as_float(optixGetPayload_2());
    const float cont_b = __uint_as_float(optixGetPayload_3());
    const uint depth = optixGetPayload_4();
    float * spectrum = group_data.spectra + spectra_index;

    // Estimate a spectrum from the RGB prior contributions
    float cont_spectrum[SPECTRAL_SAMPLES];
    makeSpectrum(make_float3(cont_r, cont_g, cont_b), cont_spectrum);
    
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
    float albedo_spectrum[SPECTRAL_SAMPLES];
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
                0.01f, 0.02f, 0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f,
                0.35f, 0.40f, 0.45f, 0.50f, 0.45f, 0.40f, 0.35f, 0.30f 
            }
        }
    };
    int light_count = sizeof(lights) / sizeof(Light);
    
    
    const float3 outgoing = -1 * optixGetWorldRayDirection();
    for (int i = 0; i < light_count; ++i) {
        for (int s = 0; s < lights[i].samples; ++s) {
            float incoming_mag;
            float3 incoming_dir;
            uint unobstructed = 0;
            if (!lights[i].sun) {
                // Select the incoming dir at random
                float3 sample = lights[i].position
                    + ((double)sobol(s, 0) / (double)__UINT32_MAX__) * lights[i].width_dir * lights[i].width
                    + ((double)sobol(s, 1) / (double)__UINT32_MAX__) * lights[i].height_dir * lights[i].height;
                float3 incoming = sample - position;
                incoming_mag = magnitude(incoming);
                incoming_dir = incoming / incoming_mag;
                optixTrace(params.handle, position, incoming_dir, 0.001, incoming_mag, 0, OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, RT_SHADOW, RT_COUNT, MT_SHADOW, unobstructed);
            } else {
                // Select the point as incoming dir
                // Remove distance from equation
                incoming_mag = 1;
                incoming_dir = lights[i].position;
                optixTrace(params.handle, position, incoming_dir, 0.001, 1e16, 0, OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, RT_SHADOW, RT_COUNT, MT_SHADOW, unobstructed);
            }
            if (!unobstructed)
                continue;
            
            // Should use the brdf to apply to spectrum
            float light_scale = lights[i].intensity * dot(normal, incoming_dir) / (incoming_mag * incoming_mag)
                / (float)lights[i].samples;
            
            
            for (int i = 0; i < SPECTRAL_SAMPLES; ++i) {
                // TODO come back to this
                /*
                spectrum[i] += light_scale * lights[i].spectrum[i]
                    * spectrumBRDF(incoming_dir, outgoing, normal, albedo_spectrum[i], group_data.roughness, group_data.metallic)
                    * cont_spectrum[i];
                */
               spectrum[i] = 1.0;
            }
        }
    }

    
    /* Indirect lighting */
    if (depth < MAX_TRACING_DEPTH - 1) {
        for (int s = 0; s < INDIRECT_SAMPLES; ++s) { 
            float3 sample_direction = sampleHemisphereUniform(normal,
                (double)sobol(s, 0) / (double)__UINT32_MAX__, (double)sobol(s, 1) / (double)__UINT32_MAX__);

            const float3 brdf = albedoBRDF(sample_direction, outgoing, normal, albedo,
                group_data.roughness, group_data.metallic);

            uint p_spectra_index = spectra_index;

            float partial = (1.0 / (float)INDIRECT_SAMPLES) * dot(sample_direction, normal);
            uint p_cont_r = __float_as_uint(cont_r * brdf.x * partial);
            uint p_cont_g = __float_as_uint(cont_g * brdf.y * partial);
            uint p_cont_b = __float_as_uint(cont_b * brdf.z * partial);
            
            uint p_next_depth = depth + 1;

            optixTrace(params.handle, position, sample_direction, 0.001, 1e16, 0, OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE, RT_RADIANCE, RT_COUNT, MT_RADIANCE, p_spectra_index, p_cont_r,
                p_cont_g, p_cont_b, p_next_depth);
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
    const uint spectra_index = optixGetPayload_0();
    const float contribution = __int_as_float(optixGetPayload_1());
    float * spectrum = params.spectra + spectra_index;

    float m_spectrum[SPECTRAL_SAMPLES];
    makeSpectrum(color, m_spectrum);

    for (int i = 0; i < SPECTRAL_SAMPLES; ++i) {
        spectrum[i] += m_spectrum[i] * contribution;
    }
}