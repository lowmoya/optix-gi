#ifndef AVA_MATH_H
#define AVA_MATH_H

#include <cuda_runtime.h>
#include <math.h>

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

/*
 * Matrix ops
 */
void inverseMatrix(float * in, float * out)
{
	float determinant = in[0] * (in[4] * in[8] - in[5] * in[7])
		- in[1] * (in[3] * in[8] - in[5] * in[6])
		+ in[2] * (in[3] * in[7] - in[4] * in[6]);
	if (!determinant) {
		fputs("Err: Calling inverse matrix with a zero-determinant input.\n",
		stderr);
		return;
	}
	out[0] = (in[4] * in[8] - in[5] * in[7]) / determinant;
	out[1] = -(in[1] * in[8] - in[2] * in[7]) / determinant;
	out[2] = (in[1] * in[5] - in[2] * in[4]) / determinant;
	out[3] = -(in[3] * in[8] - in[5] * in[6]) / determinant;
	out[4] = (in[0] * in[8] - in[2] * in[6]) / determinant;
	out[5] = -(in[0] * in[5] - in[2] * in[3]) / determinant;
	out[6] = (in[3] * in[7] - in[4] * in[6]) / determinant;
	out[7] = -(in[0] * in[7] - in[1] * in[6]) / determinant;
	out[8] = (in[0] * in[4] - in[1] * in[3]) / determinant;
}
void multMatrix(float * mat, float * v_in, float * v_out)
{
	v_out[0] = mat[0] * v_in[0] + mat[1] * v_in[1] + mat[2] * v_in[2];
	v_out[1] = mat[3] * v_in[0] + mat[4] * v_in[1] + mat[5] * v_in[2];
	v_out[2] = mat[6] * v_in[0] + mat[7] * v_in[1] + mat[8] * v_in[2];
}

/*
 * Standard ops 
 */
__device__ __host__ int min(int a, int b) { return a < b ? a : b; }
__device__ __host__ int max(int a, int b) { return a < b ? b : a; }

#endif