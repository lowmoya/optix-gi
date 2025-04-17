#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfRgba.h>
#include <Imath/ImathBox.h>

#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
		// options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
		OPTIX_CALL(optixDeviceContextCreate(cu_context, &options, &context));
	}

	/* Create a Module from PTX file. */
	OptixPipelineCompileOptions pipeline_comp_options = {
		.usesMotionBlur = false,
		.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
		.numPayloadValues = 4, // change to three when using structered payload
		.numAttributeValues = 2,
		.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
		.pipelineLaunchParamsVariableName = "params"
	};
	OptixModule radiance_module = nullptr; {
		OptixModuleCompileOptions module_options = {
			.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
			.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL
		};

		int ptx_length;
		char * ptx = readFile("build/device/radiance.ptx", ptx_length);
		
		OPTIX_CALL(optixModuleCreate(
			context, &module_options, &pipeline_comp_options, ptx, ptx_length, LOG, &LOG_SIZE, &radiance_module
		));
		free(ptx);
	}
	OptixModule shadow_module = nullptr; {
		OptixModuleCompileOptions module_options = {
			.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
			.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL
		};

		int ptx_length;
		char * ptx = readFile("build/device/shadow.ptx", ptx_length);
		
		OPTIX_CALL(optixModuleCreate(
			context, &module_options, &pipeline_comp_options, ptx, ptx_length, LOG, &LOG_SIZE, &shadow_module
		));
		free(ptx);
	}


	/* Create program groups from the Module, which correspond to specific function calls from the
	 * PTX file. */
	OptixProgramGroup radiance_raygen_program, radiance_hit_program, radiance_miss_program; {
		OptixProgramGroupOptions options = {};

		OptixProgramGroupDesc raygen_description = {
			.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
			.raygen = {
				.module = radiance_module,
				.entryFunctionName = "__raygen__radiance"
			}
		};
		OPTIX_LOG_CALL(optixProgramGroupCreate(
			context, &raygen_description, 1, &options, LOG, &LOG_SIZE, &radiance_raygen_program
		));

		OptixProgramGroupDesc miss_description = {
			.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
			.miss = {
				.module = radiance_module,
				.entryFunctionName = "__miss__radiance"
			}
		};
		OPTIX_LOG_CALL(optixProgramGroupCreate(
			context, &miss_description, 1, &options, LOG, &LOG_SIZE, &radiance_miss_program
		));

		OptixProgramGroupDesc hit_description = {
			.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
			.hitgroup = {
				.moduleCH = radiance_module,
				.entryFunctionNameCH = "__closesthit__radiance"
			}
		};
		OPTIX_LOG_CALL(optixProgramGroupCreate(
			context, &hit_description, 1, &options, LOG, &LOG_SIZE, &radiance_hit_program
		));
	}
	OptixProgramGroup shadow_hit_program, shadow_miss_program; {
		OptixProgramGroupOptions options = {};

		OptixProgramGroupDesc hit_description = {
			.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
			.hitgroup = {
				.moduleAH = shadow_module,
				.entryFunctionNameAH = "__anyhit__shadow"
			}
		};
		OPTIX_LOG_CALL(optixProgramGroupCreate(
			context, &hit_description, 1, &options, LOG, &LOG_SIZE, &shadow_hit_program
		));

		OptixProgramGroupDesc miss_description = {
			.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
			.miss = {
				.module = shadow_module,
				.entryFunctionName = "__miss__shadow"
			}
		};
		OPTIX_LOG_CALL(optixProgramGroupCreate(
			context, &miss_description, 1, &options, LOG, &LOG_SIZE, &shadow_miss_program
		));
	}

	/* Create a pipeline from the groups. */
	OptixPipeline pipeline = nullptr; {
		/* Creation. */
		OptixProgramGroup program_groups[] = { radiance_raygen_program, radiance_hit_program, radiance_miss_program, shadow_hit_program, shadow_miss_program };

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


	/* Prepare accelerated structures. */
	
	const struct Description {
		const char * path;
		float transform[12];
	} descriptions[] = {
		{"resources/floor.glb",
			{1, 0, 0, 0,	0, 1, 0, 0, 	0, 0, 1, 0}},
		{"resources/room.glb",
			{1, 0, 0, 0,	0, 1, 0, 0, 	0, 0, 1, 0}},
		{"resources/simple_dining_table.glb",
			{.004, 0, 0, 0,	0, .004, 0, .7, 	0, 0, .004, 0}},
		{"resources/retro_light.glb",
			{.01, 0, 0, 0,	0, .01, 0, 6.7, 	0, 0, .01, 0}},
		// {"resources/test.glb", make_float3(.6,.6,.6), .9, .1,
		// 	{.01, 0, 0, 1,	0, .01, 0, 5, 	0, 0, .01, 0}},
	};
	const int file_count = sizeof(descriptions) / sizeof(*descriptions);

	struct Model {
		CUdeviceptr d_index;
		CUdeviceptr d_vertex;
		CUdeviceptr d_normal;
		CUdeviceptr d_uv;
		CUdeviceptr d_gas_mem;
		int material_id;
	};
#define MAX_MODELS 60
	Model models[MAX_MODELS];
	int model_count = 0;


	struct Material {
		float roughness;
		float metallic;
		int texture_id;
	};
#define MAX_MATERIALS 60
	Material materials[MAX_MATERIALS];
	int material_count = 0;

	struct Texture {
		cudaArray_t d_array;
		cudaTextureObject_t d_texture;
		int width;
		int height;
	};
	const char * texture_paths[] = {
		"resources/sdt_fabric.png",
		"resources/sdt_marble.png",
		"resources/sdt_niunal.png",
		"resources/sdt_plant_lvzhi.png",
		"resources/sdt_qita.png",
		"resources/sdt_white_boli.png",
		"resources/grass.jpg",
		"resources/curtains.png",
		"resources/counter.jpg",
		"resources/floor.jpg",
		"resources/wall.jpg",
	};
	const int texture_count = sizeof(texture_paths) / sizeof(*texture_paths);
	Texture textures[texture_count];

	for (int i = 0; i < texture_count; ++i) {
		Texture & tex = textures[i];

		int channels;
		unsigned char * data = stbi_load(texture_paths[i], &tex.width, &tex.height, &channels, 4);
		if (!data) {
			fprintf(stderr, "Failed to load '%s',\n", texture_paths[i]);
			exit(1);
		}

		auto chan_desc = cudaCreateChannelDesc<uchar4>();
		cudaMallocArray(&tex.d_array, &chan_desc, tex.width, tex.height);
		cudaMemcpy2DToArray(tex.d_array, 0, 0, data, tex.width
			* sizeof(uchar4), tex.width * sizeof(uchar4),
			tex.height, cudaMemcpyHostToDevice);
		stbi_image_free(data);

		
		cudaResourceDesc resource_desc = {};
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = tex.d_array;

		cudaTextureDesc texture_desc = {};
		texture_desc.addressMode[0] = cudaAddressModeWrap;
		texture_desc.addressMode[1] = cudaAddressModeWrap;
		texture_desc.filterMode = cudaFilterModeLinear;
		texture_desc.readMode = cudaReadModeNormalizedFloat;
		texture_desc.normalizedCoords = 1;

		cudaCreateTextureObject(&tex.d_texture, &resource_desc, &texture_desc, 0);
	}

	CUdeviceptr d_tlas_mem;
	OptixTraversableHandle tlas_handle;  {
		const uint32_t input_flags[] = { OPTIX_GEOMETRY_FLAG_NONE };

		OptixInstance instances[MAX_MODELS];
		for (uint f = 0; f < file_count; ++f) {
			// Load scene
			Assimp::Importer importer;
			const aiScene * scene = importer.ReadFile(
				descriptions[f].path,
				aiProcess_Triangulate | aiProcess_JoinIdenticalVertices
				| aiProcess_GenNormals | aiProcess_OptimizeMeshes
				| aiProcess_PreTransformVertices | aiProcess_EmbedTextures
			);
			if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)
					|| !scene->mRootNode) {
				fprintf(stderr, "[Error] Failed to load '%s'.\n", descriptions[f].path);
				exit(1);
			}

			// Load the materials
			int other_scene_materials = material_count;
			for (int mi = 0; mi < scene->mNumMaterials; ++mi) {
				materials[material_count] = {
					.roughness = 1.0,
					.metallic = 0.0
				};
				
				aiMaterial * ref_material = scene->mMaterials[mi];
				ref_material->Get("GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR", 0, 0,
					materials[material_count].roughness);
				ref_material->Get("GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR", 0, 0,
					materials[material_count].metallic);

				materials[material_count].texture_id = 7;
				
				++material_count;
				assert(material_count != MAX_MATERIALS);
			}

			// Load each mesh into an instance
			for (int mi = 0; mi < scene->mNumMeshes; ++mi) {
				aiMesh * mesh = scene->mMeshes[mi];
				models[model_count].material_id = other_scene_materials + mesh->mMaterialIndex;

				uint vertex_count = mesh->mNumVertices, face_count = mesh->mNumFaces;
				
				// Read data
				uint3 * indices =   (uint3 *)malloc(face_count * sizeof(uint3));
				float3 * vertices = (float3 *)malloc(vertex_count * sizeof(float3));
				float3 * normals =  (float3 *)malloc(vertex_count * sizeof(float3));
				float2 * uv =  (float2 *)malloc(vertex_count * sizeof(float2));

				for (int f = 0; f < face_count; ++f) {
					assert(mesh->mFaces[f].mNumIndices == 3);
					indices[f] = make_uint3(
						mesh->mFaces[f].mIndices[0],
						mesh->mFaces[f].mIndices[1],
						mesh->mFaces[f].mIndices[2]
					);
				}

				for (int v = 0; v < vertex_count; ++v) {
					vertices[v] = make_float3(
						mesh->mVertices[v].x,
						mesh->mVertices[v].y,
						mesh->mVertices[v].z
					);
					normals[v] = make_float3(
						mesh->mNormals[v].x,
						mesh->mNormals[v].y,
						mesh->mNormals[v].z
					);
					uv[v] = make_float2(
						mesh->mTextureCoords[0][v].x,
						1.0 - mesh->mTextureCoords[0][v].y
					);
				}

				// Fill out build info
				CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&models[model_count].d_index),
					face_count * sizeof(uint3)));
				CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&models[model_count].d_vertex),
					vertex_count * sizeof(float3)));
				CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&models[model_count].d_normal),
					vertex_count * sizeof(float3)));
				CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&models[model_count].d_uv),
					vertex_count * sizeof(float2)));

				CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(models[model_count].d_index),
					indices, face_count * sizeof(uint3), cudaMemcpyHostToDevice));
				CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(models[model_count].d_vertex),
					vertices, vertex_count * sizeof(float3), cudaMemcpyHostToDevice));
				CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(models[model_count].d_normal),
					normals, vertex_count * sizeof(float3), cudaMemcpyHostToDevice));
				CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(models[model_count].d_uv),
					uv, vertex_count * sizeof(float2), cudaMemcpyHostToDevice));
				
				free(indices);
				free(vertices);
				free(normals);
				free(uv);
				
				OptixBuildInput input = {
					.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
					.triangleArray = {
						.vertexBuffers = &models[model_count].d_vertex,
						.numVertices = vertex_count,
						.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
						.indexBuffer = models[model_count].d_index,
						.numIndexTriplets = face_count,
						.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
						.flags = input_flags,
						.numSbtRecords = 1
					}
				};

				OptixAccelBuildOptions options = {
					.buildFlags = OPTIX_BUILD_FLAG_NONE,
					.operation = OPTIX_BUILD_OPERATION_BUILD
				};
				OptixAccelBufferSizes buffer_sizes;
				optixAccelComputeMemoryUsage(context, &options, &input, 1, &buffer_sizes);	
			
				CUdeviceptr d_tmp_mem;
				CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_tmp_mem), buffer_sizes.tempSizeInBytes));
				CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&models[model_count].d_gas_mem), buffer_sizes.outputSizeInBytes));
		
				// Build the structure
				OptixTraversableHandle gas_handle;
				OPTIX_CALL(optixAccelBuild(context, 0, &options, &input, 1,
					d_tmp_mem, buffer_sizes.tempSizeInBytes, models[model_count].d_gas_mem,
					buffer_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));
		
				CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_tmp_mem)));


				instances[model_count] = {
					.instanceId = 0,
					.sbtOffset = (uint)model_count,
					.visibilityMask = 255,
					.flags = OPTIX_INSTANCE_FLAG_NONE,
					.traversableHandle = gas_handle
				};
				memcpy(instances[model_count].transform, descriptions[f].transform, sizeof(instances[model_count].transform));
				
				model_count++;
				assert(model_count != MAX_MODELS);
			}
		}

		// 1 mat is floor
		// 2 room
		// materials[0].texture_id = 6;
		// materials[1].texture_id = 7;
		// materials[5].texture_id = 4; // plates / bowls
		// materials[6].texture_id = 3; // plants
		// materials[7].texture_id = 1; // table / chairs
		// materials[8].texture_id = 0; // tablemat
		// materials[9].texture_id = 4;
		// materials[0].texture_id = 6;
		// materials[1].texture_id = 7;

		materials[0].texture_id = 6; // grass
		materials[1].texture_id = 10; // wall
		materials[2].texture_id = 8; // floor
		materials[3].texture_id = 9; // counter
		materials[4].texture_id = 1; // curtain rod
		materials[5].texture_id = 7; // curtain
		materials[7].texture_id = 4; // plates / bowls
		materials[8].texture_id = 3; // plants
		materials[9].texture_id = 1; // table / chairs
		materials[10].texture_id = 0; // tablemat
		materials[11].texture_id = 4; // plates / bowls

		// Allocate memory for the build
		CUdeviceptr d_instances;
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_instances), sizeof(instances)));
		CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(d_instances), &instances, sizeof(instances),
			cudaMemcpyHostToDevice));

		OptixBuildInput tlas_input = {
			.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
			.instanceArray = {
				.instances = d_instances,
				.numInstances = (uint)model_count
			}
		};
		OptixAccelBuildOptions accel_options = {
			.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
			.operation = OPTIX_BUILD_OPERATION_BUILD
		};

		OptixAccelBufferSizes buffer_sizes;
		optixAccelComputeMemoryUsage(
			context, &accel_options, &tlas_input, 1, &buffer_sizes
		);
		CUdeviceptr d_tmp_mem;
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_tmp_mem), buffer_sizes.tempSizeInBytes));
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_tlas_mem), buffer_sizes.outputSizeInBytes));
		OPTIX_CALL(optixAccelBuild(context, 0, &accel_options, &tlas_input, 1, d_tmp_mem, buffer_sizes.tempSizeInBytes,
			d_tlas_mem, buffer_sizes.outputSizeInBytes, &tlas_handle, nullptr, 0));
		
		// Cleanup
		CUDA_CALL(cudaFree(reinterpret_cast<void*>(d_tmp_mem)));
	}


	/* Prepare environment map. */
	cudaArray_t d_sunset_array;
	cudaTextureObject_t d_sunset_texture;
	{
		int width, height, channels;
		float * data = stbi_loadf("resources/sunset.hdr", &width, &height, &channels, 4);
		if (!data) {
			fputs("Failed to load 'resources/sunset.hdr'.\n", stderr);
			exit(1);
		}
		auto chan_desc = cudaCreateChannelDesc<float4>();
		cudaMallocArray(&d_sunset_array, &chan_desc, width, height, 0);
		cudaMemcpy2DToArray(d_sunset_array, 0, 0, data, width * sizeof(float4), width * sizeof(float4),
			height, cudaMemcpyHostToDevice);
		stbi_image_free(data);


		cudaResourceDesc resource_desc = {};
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = d_sunset_array;

		cudaTextureDesc texture_desc = {};
		texture_desc.addressMode[0] = cudaAddressModeWrap;
		texture_desc.addressMode[1] = cudaAddressModeWrap;
		texture_desc.filterMode = cudaFilterModeLinear;
		texture_desc.readMode = cudaReadModeElementType;
		texture_desc.normalizedCoords = 1;

		cudaCreateTextureObject(&d_sunset_texture, &resource_desc, &texture_desc, 0);
	}

	/* Set up shader binding table. */
	// Have closest hit include the geometries index, vertex, normal info
	// they should all share a pointer to a list of area lights.
	OptixShaderBindingTable sbt = {}; {
		/* Raygen record. */
		// Create record on Host.
		SbtRecord<void> raygen_record = {};
		OPTIX_CALL(optixSbtRecordPackHeader(radiance_raygen_program, &raygen_record));

		// Allocate record on GPU and copy data.
		CUdeviceptr d_raygen_record;
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(raygen_record)));
		CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &raygen_record,
			sizeof(raygen_record), cudaMemcpyHostToDevice));


		/* Hit record. */
		// Create record on Host.
		SbtRecord<HitGroupData> hit_records[model_count * RT_COUNT] = {};
		for (int r = 0; r < RT_COUNT; ++r) {
			for (int m = 0; m < model_count; ++m) {
				const int index = m + r * model_count;
				Material & material = materials[models[m].material_id];

				switch(r) {
				case RT_RADIANCE:
					OPTIX_CALL(optixSbtRecordPackHeader(radiance_hit_program, hit_records + index));
					hit_records[index].data = {
						.indices = reinterpret_cast<uint3*>(models[m].d_index),
						.vertices = reinterpret_cast<float3*>(models[m].d_vertex),
						.normals = reinterpret_cast<float3*>(models[m].d_normal),
						.uv = reinterpret_cast<float2*>(models[m].d_uv),
						.metallic = material.metallic,
						.roughness = material.roughness,
						.texture = textures[material.texture_id].d_texture
					};

					break;
				case RT_SHADOW:
					OPTIX_CALL(optixSbtRecordPackHeader(shadow_hit_program, hit_records + index));
					hit_records[index].data = {};
					break;
				}
				
			}
		}
		
		// Allocate record on GPU and copy data.
		CUdeviceptr d_hit_records;
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_hit_records), sizeof(hit_records)));
		CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(d_hit_records), &hit_records,
			sizeof(hit_records), cudaMemcpyHostToDevice));

		/* Miss record. */
		// Create record on GPU.
		SbtRecord<MissData> miss_records[MT_COUNT] = {};
		OPTIX_CALL(optixSbtRecordPackHeader(radiance_miss_program, miss_records + MT_RADIANCE));
		miss_records[0].data.environment = d_sunset_texture;
		OPTIX_CALL(optixSbtRecordPackHeader(shadow_miss_program, miss_records + MT_SHADOW));

		// Create record on Host and copy to GPU.
		CUdeviceptr d_miss_records;
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), sizeof(miss_records)));
		CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(d_miss_records), &miss_records,
			sizeof(miss_records), cudaMemcpyHostToDevice));

		/* Assign records to the binding table. */
		sbt.raygenRecord = d_raygen_record;
		sbt.hitgroupRecordBase = d_hit_records;
		sbt.hitgroupRecordStrideInBytes = sizeof(hit_records[0]);
		sbt.hitgroupRecordCount = model_count * RT_COUNT;
		sbt.missRecordBase = d_miss_records;
		sbt.missRecordStrideInBytes = sizeof(miss_records[0]);
		sbt.missRecordCount = sizeof(miss_records)/sizeof(miss_records[0]);
	}

	uchar4 * d_render_buffer = nullptr;
	const size_t output_width = 800, output_height = 600;
	const size_t output_count = output_width * output_height;
	const size_t output_size = output_count * sizeof(uchar4);
	CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_render_buffer), output_size));


	/* Launch the application. */
	puts("Launching application.");
	{
		CUstream stream;
		CUDA_CALL(cudaStreamCreate(&stream));

		Params params;
		params.image = d_render_buffer;
		params.image_width = output_width;
		params.image_height = output_height;
		params.cam_eye = make_float3(0, 6, -6);
		params.cam_u = make_float3(1, 0, 0);
		params.cam_v = normalized(make_float3(0, 1, .1));
		params.cam_w = normalized(make_float3(0, -1, 2));
		params.handle = tlas_handle;


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
		CUDA_CALL(cudaFree(reinterpret_cast<void*>(d_tlas_mem)));
		for (int i = 0; i < model_count; ++i) {
			CUDA_CALL(cudaFree(reinterpret_cast<void *>(models[i].d_index)));
			CUDA_CALL(cudaFree(reinterpret_cast<void *>(models[i].d_vertex)));
			CUDA_CALL(cudaFree(reinterpret_cast<void *>(models[i].d_normal)));
			CUDA_CALL(cudaFree(reinterpret_cast<void *>(models[i].d_uv)));
			CUDA_CALL(cudaFree(reinterpret_cast<void *>(models[i].d_gas_mem)));
		}
		for (int i = 0; i < texture_count; ++i) {
			CUDA_CALL(cudaDestroyTextureObject(textures[i].d_texture));
			CUDA_CALL(cudaFreeArray(textures[i].d_array));
		}
		CUDA_CALL(cudaDestroyTextureObject(d_sunset_texture));
		CUDA_CALL(cudaFreeArray(d_sunset_array));

		OPTIX_CALL(optixPipelineDestroy(pipeline));
		OPTIX_CALL(optixProgramGroupDestroy(radiance_raygen_program));
		OPTIX_CALL(optixProgramGroupDestroy(radiance_hit_program));
		OPTIX_CALL(optixProgramGroupDestroy(radiance_miss_program));

		OPTIX_CALL(optixDeviceContextDestroy(context));
	}
}