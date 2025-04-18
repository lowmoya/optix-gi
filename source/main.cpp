#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <Imath/ImathBox.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
const float CIE_X[] = {
    // CIE X function values
    0.01431000,     0.01570443,     0.01714744,     0.01878122,     0.02074801,
    0.02319000,     0.02620736,     0.02978248,     0.03388092,     0.03846824,
    0.04351000,     0.04899560,     0.05502260,     0.06171880,     0.06921200,
    0.07763000,     0.08695811,     0.09717672,     0.1084063,      0.1207672,
    0.1343800,      0.1493582,      0.1653957,      0.1819831,      0.1986110,
    0.2147700,      0.2301868,      0.2448797,      0.2587773,      0.2718079,
    0.2839000,      0.2949438,      0.3048965,      0.3137873,      0.3216454,
    0.3285000,      0.3343513,      0.3392101,      0.3431213,      0.3461296,
    0.3482800,      0.3495999,      0.3501474,      0.3500130,      0.3492870,
    0.3480600,      0.3463733,      0.3442624,      0.3418088,      0.3390941,
    0.3362000,      0.3331977,      0.3300411,      0.3266357,      0.3228868,
    0.3187000,      0.3140251,      0.3088840,      0.3032904,      0.2972579,
    0.2908000,      0.2839701,      0.2767214,      0.2689178,      0.2604227,
    0.2511000,      0.2408475,      0.2298512,      0.2184072,      0.2068115,
    0.1953600,      0.1842136,      0.1733273,      0.1626881,      0.1522833,
    0.1421000,      0.1321786,      0.1225696,      0.1132752,      0.1042979,
    0.09564000,     0.08729955,     0.07930804,     0.07171776,     0.06458099,
    0.05795001,     0.05186211,     0.04628152,     0.04115088,     0.03641283,
    0.03201000,     0.02791720,     0.02414440,     0.02068700,     0.01754040,
    0.01470000,     0.01216179,     0.009919960,    0.007967240,    0.006296346,
    0.004900000,    0.003777173,    0.002945320,    0.002424880,    0.002236293,
    0.002400000,    0.002925520,    0.003836560,    0.005174840,    0.006982080,
    0.009300000,    0.01214949,     0.01553588,     0.01947752,     0.02399277,
    0.02910000,     0.03481485,     0.04112016,     0.04798504,     0.05537861,
    0.06327000,     0.07163501,     0.08046224,     0.08973996,     0.09945645,
    0.1096000,      0.1201674,      0.1311145,      0.1423679,      0.1538542,
    0.1655000,      0.1772571,      0.1891400,      0.2011694,      0.2133658,
    0.2257499,      0.2383209,      0.2510668,      0.2639922,      0.2771017,
    0.2904000,      0.3038912,      0.3175726,      0.3314384,      0.3454828,
    0.3597000,      0.3740839,      0.3886396,      0.4033784,      0.4183115,
    0.4334499,      0.4487953,      0.4643360,      0.4800640,      0.4959713,
    0.5120501,      0.5282959,      0.5446916,      0.5612094,      0.5778215,
    0.5945000,      0.6112209,      0.6279758,      0.6447602,      0.6615697,
    0.6784000,      0.6952392,      0.7120586,      0.7288284,      0.7455188,
    0.7621000,      0.7785432,      0.7948256,      0.8109264,      0.8268248,
    0.8425000,      0.8579325,      0.8730816,      0.8878944,      0.9023181,
    0.9163000,      0.9297995,      0.9427984,      0.9552776,      0.9672179,
    0.9786000,      0.9893856,      0.9995488,      1.0090892,      1.0180064,
    1.0263000,      1.0339827,      1.0409860,      1.0471880,      1.0524667,
    1.0567000,      1.0597944,      1.0617992,      1.0628068,      1.0629096,
    1.0622000,      1.0607352,      1.0584436,      1.0552244,      1.0509768,
    1.0456000,      1.0390369,      1.0313608,      1.0226662,      1.0130477,
    1.0026000,      0.9913675,      0.9793314,      0.9664916,      0.9528479,
    0.9384000,      0.9231940,      0.9072440,      0.8905020,      0.8729200,
    0.8544499,      0.8350840,      0.8149460,      0.7941860,      0.7729540,
    0.7514000,      0.7295836,      0.7075888,      0.6856022,      0.6638104,
    0.6424000,      0.6215149,      0.6011138,      0.5811052,      0.5613977,
    0.5419000,      0.5225995,      0.5035464,      0.4847436,      0.4661939,
    0.4479000,      0.4298613,      0.4120980,      0.3946440,      0.3775333,
    0.3608000,      0.3444563,      0.3285168,      0.3130192,      0.2980011,
    0.2835000,      0.2695448,      0.2561184,      0.2431896,      0.2307272,
    0.2187000,      0.2070971,      0.1959232,      0.1851708,      0.1748323,
    0.1649000,      0.1553667,      0.1462300,      0.1374900,      0.1291467,
    0.1212000,      0.1136397,      0.1064650,      0.09969044,     0.09333061,
    0.08740000,     0.08190096,     0.07680428,     0.07207712,     0.06768664,
    0.06360000,     0.05980685,     0.05628216,     0.05297104,     0.04981861,
    0.04677000,     0.04378405,     0.04087536,     0.03807264,     0.03540461,
    0.03290000,     0.03056419,     0.02838056,     0.02634484,     0.02445275,
    0.02270000,     0.02108429,     0.01959988,     0.01823732,     0.01698717,
    0.01584000,     0.01479064,     0.01383132,     0.01294868,     0.01212920,
    0.01135916};

const float CIE_Y[] = {
    // CIE Y function values
    0.0003960000,    0.0004337147,    0.0004730240,    0.0005178760,    0.0005722187,
    0.0006400000,    0.0007245600,    0.0008255000,    0.0009411600,    0.001069880,
    0.001210000,     0.001362091,     0.001530752,     0.001720368,     0.001935323,
    0.002180000,     0.002454800,     0.002764000,     0.003117800,     0.003526400,
    0.004000000,     0.004546240,     0.005159320,     0.005829280,     0.006546160,
    0.007300000,     0.008086507,     0.008908720,     0.009767680,     0.01066443,
    0.01160000,      0.01257317,      0.01358272,      0.01462968,      0.01571509,
    0.01684000,      0.01800736,      0.01921448,      0.02045392,      0.02171824,
    0.02300000,      0.02429461,      0.02561024,      0.02695857,      0.02835125,
    0.02980000,      0.03131083,      0.03288368,      0.03452112,      0.03622571,
    0.03800000,      0.03984667,      0.04176800,      0.04376600,      0.04584267,
    0.04800000,      0.05024368,      0.05257304,      0.05498056,      0.05745872,
    0.06000000,      0.06260197,      0.06527752,      0.06804208,      0.07091109,
    0.07390000,      0.07701600,      0.08026640,      0.08366680,      0.08723280,
    0.09098000,      0.09491755,      0.09904584,      0.1033674,       0.1078846,
    0.1126000,       0.1175320,       0.1226744,       0.1279928,       0.1334528,
    0.1390200,       0.1446764,       0.1504693,       0.1564619,       0.1627177,
    0.1693000,       0.1762431,       0.1835581,       0.1912735,       0.1994180,
    0.2080200,       0.2171199,       0.2267345,       0.2368571,       0.2474812,
    0.2586000,       0.2701849,       0.2822939,       0.2950505,       0.3085780,
    0.3230000,       0.3384021,       0.3546858,       0.3716986,       0.3892875,
    0.4073000,       0.4256299,       0.4443096,       0.4633944,       0.4829395,
    0.5030000,       0.5235693,       0.5445120,       0.5656900,       0.5869653,
    0.6082000,       0.6293456,       0.6503068,       0.6708752,       0.6908424,
    0.7100000,       0.7281852,       0.7454636,       0.7619694,       0.7778368,
    0.7932000,       0.8081104,       0.8224962,       0.8363068,       0.8494916,
    0.8620000,       0.8738108,       0.8849624,       0.8954936,       0.9054432,
    0.9148501,       0.9237348,       0.9320924,       0.9399226,       0.9472252,
    0.9540000,       0.9602561,       0.9660074,       0.9712606,       0.9760225,
    0.9803000,       0.9840924,       0.9874812,       0.9903128,       0.9928116,
    0.9949501,       0.9967108,       0.9980983,       0.9991120,       0.9997482,
    1.0000000,       0.9998567,       0.9993046,       0.9983255,       0.9968987,
    0.9950000,       0.9926005,       0.9897426,       0.9864444,       0.9827241,
    0.9786000,       0.9740837,       0.9691712,       0.9638568,       0.9581349,
    0.9520000,       0.9454504,       0.9384992,       0.9311628,       0.9234576,
    0.9154000,       0.9070064,       0.8982772,       0.8892048,       0.8797816,
    0.8700000,       0.8598613,       0.8493920,       0.8386220,       0.8275813,
    0.8163000,       0.8047947,       0.7930820,       0.7811920,       0.7691547,
    0.7570000,       0.7447541,       0.7324224,       0.7200036,       0.7074965,
    0.6949000,       0.6822192,       0.6694716,       0.6566744,       0.6438448,
    0.6310000,       0.6181555,       0.6053144,       0.5924756,       0.5796379,
    0.5668000,       0.5539611,       0.5411372,       0.5283528,       0.5156323,
    0.5030000,       0.4904688,       0.4780304,       0.4656776,       0.4534032,
    0.4412000,       0.4290800,       0.4170360,       0.4050320,       0.3930320,
    0.3810000,       0.3689184,       0.3568272,       0.3447768,       0.3328176,
    0.3210000,       0.3093381,       0.2978504,       0.2865936,       0.2756245,
    0.2650000,       0.2547632,       0.2448896,       0.2353344,       0.2260528,
    0.2170000,       0.2081616,       0.1995488,       0.1911552,       0.1829744,
    0.1750000,       0.1672235,       0.1596464,       0.1522776,       0.1451259,
    0.1382000,       0.1315003,       0.1250248,       0.1187792,       0.1127691,
    0.1070000,       0.1014762,       0.09618864,      0.09112296,      0.08626485,
    0.08160000,      0.07712064,      0.07282552,      0.06871008,      0.06476976,
    0.06100000,      0.05739621,      0.05395504,      0.05067376,      0.04754965,
    0.04458000,      0.04175872,      0.03908496,      0.03656384,      0.03420048,
    0.03200000,      0.02996261,      0.02807664,      0.02632936,      0.02470805,
    0.02320000,      0.02180077,      0.02050112,      0.01928108,      0.01812069,
    0.01700000,      0.01590379,      0.01483718,      0.01381068,      0.01283478,
    0.01192000,      0.01106831,      0.01027339,      0.009533311,     0.008846157,
    0.008210000,     0.007623781,     0.007085424,     0.006591476,     0.006138485,
    0.005723000,     0.005343059,     0.004995796,     0.004676404,     0.004380075,
    0.004102000};

const float CIE_Z[] = {
    // CIE Z function values
    0.06785001,     0.07448632,     0.08136156,     0.08915364,     0.09854048,
    0.1102000,      0.1246133,      0.1417017,      0.1613035,      0.1832568,
    0.2074000,      0.2336921,      0.2626114,      0.2947746,      0.3307985,
    0.3713000,      0.4162091,      0.4654642,      0.5196948,      0.5795303,
    0.6456000,      0.7184838,      0.7967133,      0.8778459,      0.9594390,
    1.0390501,      1.1153673,      1.1884971,      1.2581233,      1.3239296,
    1.3856000,      1.4426352,      1.4948035,      1.5421903,      1.5848807,
    1.6229600,      1.6564048,      1.6852959,      1.7098745,      1.7303821,
    1.7470600,      1.7600446,      1.7696233,      1.7762637,      1.7804334,
    1.7826000,      1.7829682,      1.7816998,      1.7791982,      1.7758671,
    1.7721100,      1.7682589,      1.7640390,      1.7589438,      1.7524663,
    1.7441000,      1.7335595,      1.7208581,      1.7059369,      1.6887372,
    1.6692000,      1.6475287,      1.6234127,      1.5960223,      1.5645280,
    1.5281000,      1.4861114,      1.4395215,      1.3898799,      1.3387362,
    1.2876400,      1.2374223,      1.1878243,      1.1387611,      1.0901480,
    1.0419000,      0.9941976,      0.9473473,      0.9014531,      0.8566193,
    0.8129501,      0.7705173,      0.7294448,      0.6899136,      0.6521049,
    0.6162000,      0.5823286,      0.5504162,      0.5203376,      0.4919673,
    0.4651800,      0.4399246,      0.4161836,      0.3938822,      0.3729459,
    0.3533000,      0.3348578,      0.3175521,      0.3013375,      0.2861686,
    0.2720000,      0.2588171,      0.2464838,      0.2347718,      0.2234533,
    0.2123000,      0.2011692,      0.1901196,      0.1792254,      0.1685608,
    0.1582000,      0.1481383,      0.1383758,      0.1289942,      0.1200751,
    0.1117000,      0.1039048,      0.09666748,     0.08998272,     0.08384531,
    0.07824999,     0.07320899,     0.06867816,     0.06456784,     0.06078835,
    0.05725001,     0.05390435,     0.05074664,     0.04775276,     0.04489859,
    0.04216000,     0.03950728,     0.03693564,     0.03445836,     0.03208872,
    0.02984000,     0.02771181,     0.02569444,     0.02378716,     0.02198925,
    0.02030000,     0.01871805,     0.01724036,     0.01586364,     0.01458461,
    0.01340000,     0.01230723,     0.01130188,     0.01037792,     0.009529306,
    0.008749999,    0.008035200,    0.007381600,    0.006785400,    0.006242800,
    0.005749999,    0.005303600,    0.004899800,    0.004534200,    0.004202400,
    0.003900000,    0.003623200,    0.003370600,    0.003141400,    0.002934800,
    0.002749999,    0.002585200,    0.002438600,    0.002309400,    0.002196800,
    0.002100000,    0.002017733,    0.001948200,    0.001889800,    0.001840933,
    0.001800000,    0.001766267,    0.001737800,    0.001711200,    0.001683067,
    0.001650001,    0.001610133,    0.001564400,    0.001513600,    0.001458533,
    0.001400000,    0.001336667,    0.001270000,    0.001205000,    0.001146667,
    0.001100000,    0.001068800,    0.001049400,    0.001035600,    0.001021200,
    0.001000000,    0.0009686400,   0.0009299200,   0.0008868800,   0.0008425600,
    0.0008000000,   0.0007609600,   0.0007236800,   0.0006859200,   0.0006454400,
    0.0006000000,   0.0005478667,   0.0004916000,   0.0004354000,   0.0003834667,
    0.0003400000,   0.0003072533,   0.0002831600,   0.0002654400,   0.0002518133,
    0.0002400000,   0.0002295467,   0.0002206400,   0.0002119600,   0.0002021867,
    0.0001900000,   0.0001742133,   0.0001556400,   0.0001359600,   0.0001168533,
    0.0001000000,   0.00008613333,  0.00007460000,  0.00006500000,  0.00005693333,
    0.00004999999,  0.00004416000,  0.00003948000,  0.00003572000,  0.00003264000,
    0.00003000000,  0.00002765333,  0.00002556000,  0.00002364000,  0.00002181333,
    0.00002000000,  0.00001813333,  0.00001620000,  0.00001420000,  0.00001213333,
    0.00001000000,  0.000007733333, 0.000005400000, 0.000003200000, 0.000001333333,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000};



// Simple matrix helper functions
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
int min(int a, int b) {
	return a < b ? a : b;
}
float maxf(float a, float b) {
	return a < b ? b : a;
}

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



float gamma(float color) {
	return color <= 0.0031308 ? 12.92 * color : 1.055 * pow(color, 1.0/2.4) - 0.055;
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
		.numPayloadValues = 5,
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
		"resources/lamp.png",
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
		materials[11].texture_id = 1; // plates / bowls
		materials[14].texture_id = 11; // lamp

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


	CUdeviceptr d_spectral_buffer;
	const size_t output_width = 1000, output_height = 1000;
	const size_t output_count = output_width * output_height;
	const size_t output_size = output_count * SPECTRAL_SAMPLES * sizeof(float);
	CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_spectral_buffer), output_size));

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
					hit_records[index].data = {
					};
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


	/* Launch the application. */

	float3 cam_up = make_float3(0, 1, 0);
	// float3 cam_pos = make_float3(0, 6, -6);
	// float3 cam_w = normalized(make_float3(0, -1, 2));
	// float3 cam_pos = make_float3(4, 5, 2);
	// float3 cam_w = normalized(make_float3(-2, -1, -1));
	float3 cam_pos = make_float3(-6, 6, 0);
	float3 cam_w = normalized(make_float3(2, -1, 0));

	float3 cam_u = normalized(cross(cam_up, cam_w));
	float3 cam_v = cross(cam_w, cam_u);
	
	puts("Launching application.");
	for (uint y = 0; y < output_height; y += TILE_SIZE) {
		for (uint x = 0; x < output_width; x += TILE_SIZE) {
			CUstream stream;
			CUDA_CALL(cudaStreamCreate(&stream));

			Params params = {
				.spectra = reinterpret_cast<float *>(d_spectral_buffer),
				.output_width = output_width,
				.output_height = output_height,
				.offset_x = x,
				.offset_y = y,
				.cam_pos = cam_pos,
				.cam_u = cam_u,
				.cam_v = cam_v,
				.cam_w = cam_w,
				.handle = tlas_handle
			};


			CUdeviceptr d_params;
			CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(params)));
			CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(params),
				cudaMemcpyHostToDevice));

			OPTIX_CALL(optixLaunch(pipeline, stream, d_params, sizeof(params), &sbt,
				min(output_width - x, TILE_SIZE),
				min(output_height - y, TILE_SIZE),
				1));

			CUDA_CALL(cudaFree(reinterpret_cast<void*>(d_params)));
		}

	}

	/* Write results. */
	puts("Writing results.");
	{
		float * spectral_buffer = (float *)malloc(output_count * SPECTRAL_SAMPLES * sizeof(float));
		CUDA_CALL(cudaMemcpy(spectral_buffer, reinterpret_cast<void*>(d_spectral_buffer), output_size, cudaMemcpyDeviceToHost));
		float average = 0.0;
		for (int i = 0; i < output_count * SPECTRAL_SAMPLES; ++i) {
			average += log(maxf(spectral_buffer[i], 0.00001)) / (float)(output_count * SPECTRAL_SAMPLES);
		}
		float factor = 0.00001 / exp(average);
		for (int i = 0; i < output_count * SPECTRAL_SAMPLES; ++i)
			spectral_buffer[i] *= factor;
		// float max = 0.0;
		// for (int i = 0; i < output_count * SPECTRAL_SAMPLES; ++i) {
		// 	if (spectral_buffer[i] > max)
		// 		max = spectral_buffer[i];
		// }
		// float factor = max > 0.18 ? 1.0 / max : 1.0;
		// for (int i = 0; i < output_count * SPECTRAL_SAMPLES; ++i)
		// 	spectral_buffer[i] *= factor;

		uchar4 * output_buffer = (uchar4 *)malloc(output_count * sizeof(uchar4));

		static float xyz_transform[9];
		// Adobe RGB values
		const float xw = 0.3127, yw = 0.3290, Yw = 1.0;
		const float xr = 0.640, yr = 0.330, xg = 0.21, yg = 0.71, xb = 0.150, yb = 0.060;

		// Calculate the whitepoint colorspace.
		const float zr = 1 - xr - yr, zg = 1 - xg - yg, zb = 1 - xb - yb, zw = 1 - xw - yw;
		const float Xw = xw * Yw / yw, Zw = zw * Yw / yw;
		// Calculate the RGB channels.
		float channel_transform[9] = { xr, xg, xb, yr, yg, yb, zr, zg, zb };
		float whitepoint_transform[9];
		inverseMatrix(channel_transform, whitepoint_transform);
		float whitepoint[3] = { Xw, Yw, Zw };
		float channels[3];
		multMatrix(whitepoint_transform, whitepoint, channels);
		// Calculate the transform
		float rgb_transform[9] = {
			xr * channels[0], xg * channels[1], xb * channels[2],
			yr * channels[0], yg * channels[1], yb * channels[2],
			zr * channels[0], zg * channels[1], zb * channels[2]};
		inverseMatrix(rgb_transform, xyz_transform);

		for (int i = 0; i < output_count; ++i) {
			float * spectrum = spectral_buffer + i * SPECTRAL_SAMPLES;

			// Calculate the CIE primary values
			float X = 0, Y = 0, Z = 0;
			for (int nm = 0; nm < SPECTRAL_SAMPLES; ++nm) {
				X += spectrum[nm] * CIE_X[nm * SPECTRAL_STEP] * SPECTRAL_STEP;
				Y += spectrum[nm] * CIE_Y[nm * SPECTRAL_STEP] * SPECTRAL_STEP;
				Z += spectrum[nm] * CIE_Z[nm * SPECTRAL_STEP] * SPECTRAL_STEP;
			}

			// Calculate the final RGB values
			float xyz[3] = { X, Y, Z };
			float rgb[3];
			multMatrix(xyz_transform, xyz, rgb);

			rgb[0] = gamma(rgb[0]);
			rgb[1] = gamma(rgb[1]);
			rgb[2] = gamma(rgb[2]);

			output_buffer[i] = make_uchar4(
				min(rgb[0] * 255, 255), min(rgb[1] * 255, 255), min(rgb[2] * 255, 255), 255
			);
		}

		stbi_flip_vertically_on_write(true);
		stbi_write_png("output.png", output_width, output_height, 4, output_buffer,
			output_width * sizeof(uchar4));
		CUDA_CALL(cudaFree(reinterpret_cast<void*>(d_spectral_buffer)));
		free(output_buffer);
		free(spectral_buffer);
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