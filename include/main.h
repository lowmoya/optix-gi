#ifndef AVA_MAIN_H
#define AVA_MAIN_H

#define SPECTRAL_SAMPLES 16
#define SPECTRAL_START 400
#define SPECTRAL_END 700
#define SPECTRAL_STEP ((SPECTRAL_END - SPECTRAL_END) / (SPECTRAL_SAMPLES - 1))

#define MAX_TRACING_DEPTH 2
#define INDIRECT_SAMPLES 1
#define LIGHT_SAMPLES 1

struct Params {
    float * spectra;
    unsigned int output_width;
    unsigned int output_height;
    float3 cam_eye;
    float3 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};

struct HitGroupData {
    float * spectra;
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


/* Math extensions */
inline __host__ __device__
float2 operator*(float scalar, const float2& a) {
    return make_float2(scalar * a.x, scalar * a.y);
}
inline __host__ __device__
float2 operator*(const float2& a, float scalar) {
    return make_float2(scalar * a.x, scalar * a.y);
}
inline __host__ __device__
float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__
float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__
float3& operator+=(float3& a, float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
inline __host__ __device__
float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__
float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__
float3 operator*(const float3& a, float scalar) {
    return make_float3(scalar * a.x, scalar * a.y, scalar * a.z);
}
inline __host__ __device__
float3 operator*(float scalar, const float3& a) {
    return make_float3(scalar * a.x, scalar * a.y, scalar * a.z);
}
inline __host__ __device__
float3 operator/(const float3& a, float scalar) {
    const float ratio = 1.0 / scalar;
    return a * ratio;
}
inline __host__ __device__
float magnitude(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}
inline __host__ __device__
float3 normalized(const float3& v) {
    return v / magnitude(v);
}
inline __host__ __device__
float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__
float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, -(a.x * b.z - a.z * b.x), a.x * b.y - a.y * b.x);
}

#endif