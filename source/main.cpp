#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "main.h"

#define OPTIX_CALL(call) { if (call != OPTIX_SUCCESS) {\
		fprintf(stderr, "[Error] %s at %s:%d, returned %d\n", #call, __FILE__, __LINE__, call);\
		exit(1);\
	}}
char LOG[2048];
size_t LOG_SIZE = sizeof(LOG);
#define OPTIX_LOG_CALL(call) { if (call != OPTIX_SUCCESS) {\
		fprintf(stderr, "[Error] %s at %s:%d, returned %d\n", #call, __FILE__, __LINE__, call);\
		fprintf(stderr, "%s\n", LOG);\
		exit(1);\
	}}

#define CUDA_CALL(call) { if (call != cudaSuccess) {\
		fprintf(stderr, "[Error] %s at %s:%d, returned %s\n", #call, __FILE__, __LINE__,\
			cudaGetErrorString(call));\
		exit(1);\
	}}

template <typename T> struct SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};
template <> struct SbtRecord<void> {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

#define MAX_TRACING_DEPTH 2



char * readFile(char const * path, int & length)
{
	FILE * file = fopen(path, "r");
	if (!file) {
		length = 0;
		return nullptr;
	}

	fseek(file, 0, SEEK_END);
	length = ftell(file);
	rewind(file);

	char * buffer = (char *)malloc(length + 1);
	fread(buffer, 1, length, file);
	buffer[length] = '\0';

	fclose(file);
	return buffer;
}

int main()
{
	puts("Setting up application.");
	/* Cuda and Optix initialization. */
	CUDA_CALL(cudaFree(0));
	OPTIX_CALL(optixInit());
	CUcontext cu_context = 0;
	OptixDeviceContext context = nullptr; {
		OptixDeviceContextOptions options = {};
		options.logCallbackFunction = [](unsigned int level, const char * tag, const char * message, void*) {
			printf("[(%d)%s]\t\t%s\n", level, tag, message);
		};
		options.logCallbackLevel = 3;
		OPTIX_CALL(optixDeviceContextCreate(cu_context, &options, &context));
	}

	/* Create a Module from PTX file. */
	OptixPipelineCompileOptions pipeline_comp_options = {
		.usesMotionBlur = false,
		.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
		.numPayloadValues = 2,
		.numAttributeValues = 2,
		.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
		.pipelineLaunchParamsVariableName = "params"
	};
	OptixModule module = nullptr; {
		OptixModuleCompileOptions module_options = {
			.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
			.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL
		};

		int ptx_length;
		char * ptx = readFile("build/device/draw_solid_color.ptx", ptx_length);
		
		OPTIX_CALL(optixModuleCreate(
			context, &module_options, &pipeline_comp_options, ptx, ptx_length, LOG, &LOG_SIZE, &module
		));
		free(ptx);
	}
	/* Create program groups from the Module, which correspond to specific function calls from the
	 * PTX file. */
	OptixProgramGroup raygen_program_group = nullptr, miss_program_group = nullptr; {
		OptixProgramGroupOptions options = {};

		OptixProgramGroupDesc raygen_description = {
			.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
			.raygen = {
				.module = module,
				.entryFunctionName = "__raygen__draw_solid_color"
			}
		};
		OPTIX_LOG_CALL(optixProgramGroupCreate(
			context, &raygen_description, 1, &options, LOG, &LOG_SIZE, &raygen_program_group
		));

		OptixProgramGroupDesc miss_description = { .kind = OPTIX_PROGRAM_GROUP_KIND_MISS };
		OPTIX_LOG_CALL(optixProgramGroupCreate(
			context, &miss_description, 1, &options, LOG, &LOG_SIZE, &miss_program_group
		));
	}

	/* Create a pipeline from the groups. */
	OptixPipeline pipeline = nullptr; {
		/* Creation. */
		OptixProgramGroup program_groups[] = { raygen_program_group };

		OptixPipelineLinkOptions link_options = { .maxTraceDepth = MAX_TRACING_DEPTH };
		OPTIX_LOG_CALL(optixPipelineCreate(
			context, &pipeline_comp_options, &link_options, program_groups,
			sizeof(program_groups) / sizeof(*program_groups), LOG, &LOG_SIZE, &pipeline
		));

		/* Setup. */
		OptixStackSizes stack_sizes = {};
		for (auto & group : program_groups) {
			OPTIX_CALL(optixUtilAccumulateStackSizes(group, &stack_sizes, pipeline));
		}

		uint32_t traversal_stack_size, state_stack_size, continuation_stack_size;
		OPTIX_CALL(optixUtilComputeStackSizes(&stack_sizes, MAX_TRACING_DEPTH, 0, 0,
			&traversal_stack_size, &state_stack_size, &continuation_stack_size));
		OPTIX_CALL(optixPipelineSetStackSize(pipeline, traversal_stack_size,
			state_stack_size, continuation_stack_size, 2));
	}

	/* Set up shader binding table. */
	OptixShaderBindingTable sbt = {}; {
		/* Raygen record. */
		// Create record on Host.
		SbtRecord<RayGenData> raygen_record = { .data = {.1f, .7f, .2f} };
		OPTIX_CALL(optixSbtRecordPackHeader(raygen_program_group, &raygen_record));

		// Allocate record on GPU and copy data.
		CUdeviceptr d_raygen_record;
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(raygen_record)));
		CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &raygen_record,
			sizeof(raygen_record), cudaMemcpyHostToDevice));

		/* Miss record. */
		// Create record on GPU.
		SbtRecord<void> miss_record = {};
		OPTIX_CALL(optixSbtRecordPackHeader(miss_program_group, &miss_record))

		// Create record on Host and copy to GPU.
		CUdeviceptr d_miss_record;
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof(miss_record)));
		CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(d_miss_record), &miss_record,
			sizeof(miss_record), cudaMemcpyHostToDevice));

		/* Assign records to the binding table. */
		sbt.raygenRecord = d_raygen_record;
		sbt.missRecordBase = d_miss_record;
		sbt.missRecordStrideInBytes = sizeof(miss_record);
		sbt.missRecordCount = 1;
	}

	uchar4 * d_output_buffer = nullptr;
	const size_t output_width = 800, output_height = 600;
	const size_t output_count = output_width * output_height;
	const size_t output_size = output_count * sizeof(uchar4);
	CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_output_buffer), output_size));

	/* Launch the application. */
	puts("Launching application.");
	{
		CUstream stream;
		CUDA_CALL(cudaStreamCreate(&stream));

		Params params;
		params.image = d_output_buffer;
		params.image_width = output_width;

		CUdeviceptr d_params;
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(params)));
		CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(params),
			cudaMemcpyHostToDevice));

		OPTIX_CALL(optixLaunch(pipeline, stream, d_params, sizeof(params), &sbt,
			output_width, output_height, 1));

		CUDA_CALL(cudaFree(reinterpret_cast<void*>(d_params)));

	}

	/* Write results. */
	puts("Writing results.");
	{
		uchar4 output_buffer[output_count];
		CUDA_CALL(cudaMemcpy(output_buffer, d_output_buffer, output_size, cudaMemcpyDeviceToHost));
		stbi_write_png("output.png", output_width, output_height, 4, output_buffer,
			output_width * sizeof(uchar4));
		CUDA_CALL(cudaFree(d_output_buffer));
	}

	/* Cleanup. */
	{
		CUDA_CALL(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
		CUDA_CALL(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));

		OPTIX_CALL(optixPipelineDestroy(pipeline));
		OPTIX_CALL(optixProgramGroupDestroy(raygen_program_group));
		OPTIX_CALL(optixProgramGroupDestroy(miss_program_group));

		OPTIX_CALL(optixDeviceContextDestroy(context));
	}
}