#ifndef AVA_MATH_H
#define AVA_MATH_H

#include <math.h>
#include <stdio.h>


/*
 * Matrix ops
 */
void inverseMatrix(float * in, float * out);
void multMatrix(float * mat, float * v_in, float * v_out);

/*
 * Standard ops 
 */
__device__ __host__ int min(int a, int b);
__device__ __host__ int max(int a, int b);

/*
 * Float vec operations.
 */
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