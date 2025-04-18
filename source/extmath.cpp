#include <cuda_runtime.h>
#include "extmath.h"

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