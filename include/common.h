// Host & device definitions

#ifndef AVA_COMMON_H
#define AVA_COMMON_H

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

/*
 * Render parameters
 */
#define SPECTRAL_SAMPLES 16
#define SPECTRAL_START 400
#define SPECTRAL_END 700
#define SPECTRAL_STEP (int)((SPECTRAL_END - SPECTRAL_START) / (SPECTRAL_SAMPLES - 1))

#define MAX_TRACING_DEPTH 3
// #define INDIRECT_SAMPLES 300
// #define LIGHT_SAMPLES 40
#define INDIRECT_SAMPLES 1
#define LIGHT_SAMPLES 1

#define TILE_SIZE 1000


/*
 * Render types
 */
struct Params {
    float * spectra;
    unsigned int output_width;
    unsigned int output_height;
    unsigned int offset_x;
    unsigned int offset_y;
    float3 cam_pos;
    float3 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};

struct HitGroupData {
    uint3 * indices;
    float3 * vertices;
    float3 * normals;
    float2 * uv;
    float metallic;
    float roughness;
    cudaTextureObject_t texture;
};

struct MissData {
    cudaTextureObject_t environment;
};

enum RayType {
    RT_RADIANCE = 0,
    RT_SHADOW,
    RT_COUNT
};

enum MissType {
    MT_RADIANCE = 0,
    MT_SHADOW,
    MT_COUNT
};

#endif