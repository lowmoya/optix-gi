#ifndef AVA_MAIN_H
#define AVA_MAIN_H

// Debug wrappers
char LOG[2048];
size_t LOG_SIZE = sizeof(LOG);
#define OPTIX_CALL(call) { if (call != OPTIX_SUCCESS) {\
		fprintf(stderr, "[Error] %s:%d, %s returned %d\n", __FILE__, __LINE__, #call, call);\
		exit(1);\
	}}
#define OPTIX_LOG_CALL(call) { if (call != OPTIX_SUCCESS) {\
		fprintf(stderr, "[Error] %s:%d, %s returned %d\n", __FILE__, __LINE__, #call, call);\
		fprintf(stderr, "%s\n", LOG);\
		exit(1);\
	}}
#define CUDA_CALL(call) { if (call != cudaSuccess) {\
		fprintf(stderr, "[Error] %s:%d, %s returned %s\n", __FILE__, __LINE__, #call,\
			cudaGetErrorString(call));\
		exit(1);\
	}}

// SBT def
template <typename T> struct SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};
template <> struct SbtRecord<void> {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

#endif