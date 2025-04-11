#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "main.h"

#define OPTIX_CALL(call) { if (call != OPTIX_SUCCESS) {\
		fprintf(stderr, "[Error] %s:%d, %s returned %d\n", __FILE__, __LINE__, #call, call);\
		exit(1);\
	}}
char LOG[2048];
size_t LOG_SIZE = sizeof(LOG);
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
		.numPayloadValues = 3, // change to three when using structered payload
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
		char * ptx = readFile("build/device/main.ptx", ptx_length);
		
		OPTIX_CALL(optixModuleCreate(
			context, &module_options, &pipeline_comp_options, ptx, ptx_length, LOG, &LOG_SIZE, &module
		));
		free(ptx);
	}

	/* Create program groups from the Module, which correspond to specific function calls from the
	 * PTX file. */
	OptixProgramGroup raygen_program_group, hit_program_group, miss_program_group; {
		OptixProgramGroupOptions options = {};

		OptixProgramGroupDesc raygen_description = {
			.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
			.raygen = {
				.module = module,
				.entryFunctionName = "__raygen__gi"
			}
		};
		OPTIX_LOG_CALL(optixProgramGroupCreate(
			context, &raygen_description, 1, &options, LOG, &LOG_SIZE, &raygen_program_group
		));

		OptixProgramGroupDesc miss_description = {
			.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
			.miss = {
				.module = module,
				.entryFunctionName = "__miss__gi"
			}
		};
		OPTIX_LOG_CALL(optixProgramGroupCreate(
			context, &miss_description, 1, &options, LOG, &LOG_SIZE, &miss_program_group
		));

		OptixProgramGroupDesc hit_description = {
			.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
			.hitgroup = {
				.moduleCH = module,
				.entryFunctionNameCH = "__closesthit__gi"
			}
		};
		OPTIX_LOG_CALL(optixProgramGroupCreate(
			context, &hit_description, 1, &options, LOG, &LOG_SIZE, &hit_program_group
		));
	}

	/* Create a pipeline from the groups. */
	OptixPipeline pipeline = nullptr; {
		/* Creation. */
		OptixProgramGroup program_groups[] = { raygen_program_group, hit_program_group, miss_program_group };

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
	// Have closest hit include the geometries index, vertex, normal info
	// they should all share a pointer to a list of area lights.
	OptixShaderBindingTable sbt = {}; {
		/* Raygen record. */
		// Create record on Host.
		SbtRecord<void> raygen_record = {};
		OPTIX_CALL(optixSbtRecordPackHeader(raygen_program_group, &raygen_record));

		// Allocate record on GPU and copy data.
		CUdeviceptr d_raygen_record;
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(raygen_record)));
		CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &raygen_record,
			sizeof(raygen_record), cudaMemcpyHostToDevice));


		/* Hit record. */
		// Create record on Host.
		SbtRecord<void> hit_record = {};
		OPTIX_CALL(optixSbtRecordPackHeader(hit_program_group, &hit_record));

		// Allocate record on GPU and copy data.
		CUdeviceptr d_hit_record;
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_hit_record), sizeof(hit_record)));
		CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(d_hit_record), &hit_record, 1,
			cudaMemcpyHostToDevice));

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
		sbt.hitgroupRecordBase = d_hit_record;
		sbt.hitgroupRecordStrideInBytes = sizeof(hit_record);
		sbt.hitgroupRecordCount = 1;
		sbt.missRecordBase = d_miss_record;
		sbt.missRecordStrideInBytes = sizeof(miss_record);
		sbt.missRecordCount = 1;
	}

	uchar4 * d_render_buffer = nullptr;
	const size_t output_width = 800, output_height = 600;
	const size_t output_count = output_width * output_height;
	const size_t output_size = output_count * sizeof(uchar4);
	CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_render_buffer), output_size));

	/* Prepare accelerated structures. */
	CUdeviceptr d_traversable_mem;
	OptixTraversableHandle traversable_handle;

	const char * files[] = {
		"resources/teapot.obj",
		"resources/floor.obj"
	};
	const int model_count = sizeof(files) / sizeof(*files);

	struct {
		CUdeviceptr index;
		CUdeviceptr vertex;
	} models[model_count];

	{
		const uint32_t input_flags[] = { OPTIX_GEOMETRY_FLAG_NONE };

		OptixBuildInput inputs[model_count];

		for (int m = 0; m < model_count; ++m) {
			// Load scene
			Assimp::Importer importer;
			const aiScene * scene = importer.ReadFile(
				files[m],
				aiProcess_Triangulate | aiProcess_JoinIdenticalVertices
				| aiProcess_GenNormals | aiProcess_OptimizeMeshes
				| aiProcess_PreTransformVertices
			);
			if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)
					|| !scene->mRootNode) {
				fprintf(stderr, "[Error] Failed to load '%s'.\n", files[m]);
				exit(1);
			}

			// Measure counts
			uint vertex_count = 0, face_count = 0;
			for (int mi = 0; mi < scene->mNumMeshes; ++mi) {
				aiMesh * mesh = scene->mMeshes[mi];
				vertex_count += mesh->mNumVertices;

				for (int fi = 0; fi < mesh->mNumFaces; ++fi)
					if (mesh->mFaces[fi].mNumIndices == 3)
						++face_count;
			}

			// Read data
			float3 * vertices = (float3 *)malloc(vertex_count * sizeof(float3));
			uint * indices = (uint *)malloc(face_count * 3 * sizeof(uint));
			uint vertex_offset = 0, index_offset = 0;
			for (int mi = 0; mi < scene->mNumMeshes; ++mi) {
				aiMesh * mesh = scene->mMeshes[mi];
				memcpy(vertices + vertex_offset, mesh->mVertices, mesh->mNumVertices * sizeof(float3));
				vertex_offset += mesh->mNumVertices;

				for (int fi = 0; fi < mesh->mNumFaces; ++fi) {
					aiFace * face = mesh->mFaces + fi;
					if (face->mNumIndices == 3) {
						memcpy(indices + index_offset, face->mIndices, 3 * sizeof(uint));
						index_offset += 3;
					}
				}
			}

			// Fill out build info

			CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&models[m].index),
				face_count * sizeof(uint3)));
			CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&models[m].vertex),
				vertex_count * sizeof(float3)));

			CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(models[m].index),
				indices, face_count * sizeof(uint3), cudaMemcpyHostToDevice));
			CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(models[m].vertex),
				vertices, vertex_count * sizeof(float3), cudaMemcpyHostToDevice));
			
			free(indices);
			free(vertices);
			
			inputs[m] = {
				.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
				.triangleArray = {
					.vertexBuffers = &models[m].vertex,
					.numVertices = vertex_count,
					.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
					.indexBuffer = models[m].index,
					.numIndexTriplets = face_count,
					.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
					.flags = input_flags,
					.numSbtRecords = 1
				}
			};
		}

		// Allocate memory for the build
		OptixAccelBuildOptions options = {
			.buildFlags = OPTIX_BUILD_FLAG_NONE,
			.operation = OPTIX_BUILD_OPERATION_BUILD
		};
		OptixAccelBufferSizes buffer_sizes;
		optixAccelComputeMemoryUsage(context, &options, inputs, model_count, &buffer_sizes);	
	
		CUdeviceptr d_temp_mem;
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_temp_mem), buffer_sizes.tempSizeInBytes));
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_traversable_mem), buffer_sizes.outputSizeInBytes));

		// Build the structure
		OPTIX_CALL(optixAccelBuild(context, 0, &options, inputs, model_count,
			d_temp_mem, buffer_sizes.tempSizeInBytes, d_traversable_mem,
			buffer_sizes.outputSizeInBytes, &traversable_handle, nullptr, 0));

		// Cleanup
		CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_temp_mem)));
	}

	/* Launch the application. */
	puts("Launching application.");
	{
		CUstream stream;
		CUDA_CALL(cudaStreamCreate(&stream));

		Params params;
		params.image = d_render_buffer;
		params.image_width = output_width;
		params.image_height = output_height;
		params.cam_eye = make_float3(0, 4, -6);
		params.cam_u = make_float3(1, 0, 0);
		params.cam_v = make_float3(0, 1, 0);
		params.cam_w = normalized(make_float3(0, -1, 2));
		params.handle = traversable_handle;


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
		CUDA_CALL(cudaMemcpy(output_buffer, d_render_buffer, output_size, cudaMemcpyDeviceToHost));
		stbi_flip_vertically_on_write(true);
		stbi_write_png("output.png", output_width, output_height, 4, output_buffer,
			output_width * sizeof(uchar4));
		CUDA_CALL(cudaFree(d_render_buffer));
	}

	/* Cleanup. */
	{
		CUDA_CALL(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
		CUDA_CALL(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
		CUDA_CALL(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
		CUDA_CALL(cudaFree(reinterpret_cast<void*>(d_traversable_mem)));
		for (auto& model : models) {
			CUDA_CALL(cudaFree(reinterpret_cast<void *>(model.index)));
			CUDA_CALL(cudaFree(reinterpret_cast<void *>(model.vertex)));
		}

		OPTIX_CALL(optixPipelineDestroy(pipeline));
		OPTIX_CALL(optixProgramGroupDestroy(raygen_program_group));
		OPTIX_CALL(optixProgramGroupDestroy(hit_program_group));
		OPTIX_CALL(optixProgramGroupDestroy(miss_program_group));

		OPTIX_CALL(optixDeviceContextDestroy(context));
	}
}