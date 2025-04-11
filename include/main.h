#ifndef AVA_MAIN_H
#define AVA_MAIN_H

struct Params {
    uchar4 * image;
    unsigned int image_width;
    unsigned int image_height;
    float3 cam_eye;
    float3 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};

/* Math extensions */
inline __host__ __device__
float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
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

#endif