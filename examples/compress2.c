#include <stdint.h>
#include <dlfcn.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include "xmmintrin.h"
#include <emmintrin.h>
#include <x86intrin.h>
#include <strings.h> /* for bzero */
#include "time.h"
#define _GNU_SOURCE
#define __USE_GNU
#include <sched.h>  /* for CPU_ZERO etc */

#include <pthread.h> /* for pthread_setaffinity_np */

#define BFPC_9BIT 9
#define BFPC_12BIT 12

// for PCM
#ifdef ENABLE_PCM
int pcm_getcpu()
{
	int id = -1;
	asm volatile (
		"rdtscp\n\t"
		"mov %%ecx, %0\n\t":
		"=r" (id) :: "%rax", "%rcx", "%rdx");
	// processor ID is in ECX: https://www.felixcloutier.com/x86/rdtscp
	// Linux encodes the NUMA node starting at bit 12, so remove the NUMA
	// bits when returning the CPU integer by masking with 0xFFF.
	return id & 0xFFF;
}

/* for pcm communication */
#define NB_PIPE_FD 2
#define READ_PIPE_FD 0
#define WRITE_PIPE_FD 1
int pipefd[NB_PIPE_FD];

struct {
	int (*pcm_c_build_core_event)(uint8_t id, const char * argv);
	int (*pcm_c_init)();
	void (*pcm_c_start)();
	void (*pcm_c_stop)();
	uint64_t (*pcm_c_get_cycles)(uint32_t core_id);
	uint64_t (*pcm_c_get_instr)(uint32_t core_id);
	uint64_t (*pcm_c_get_core_event)(uint32_t core_id, uint32_t event_id);
} PCM;


#ifndef PCM_DYNAMIC_LIB
/* Library functions declaration (instead of .h file) */
int pcm_c_build_core_event(uint8_t, const char *);
int pcm_c_init();
void pcm_c_start();
void pcm_c_stop();
uint64_t pcm_c_get_cycles(uint32_t);
uint64_t pcm_c_get_instr(uint32_t);
uint64_t pcm_c_get_core_event(uint32_t, uint32_t);
#endif
#endif

// BFP Compression Request
struct xranlib_compress_request {
    int16_t *data_in;   /*!< Pointer to data to compress. */
    int16_t numRBs;     /*!< numRBs  */
    int16_t numDataElements; /*!< number of elements in block process [UP: 24 i.e 12RE*2] */
    int16_t iqWidth;    /*!< Bit size */
    int len;        /*!< Length of input buffer in bytes */
    int bfpc_exp_offset; // bfpcExpOffSet;
};

// BFP Compression Response
struct xranlib_compress_response {
    int8_t *data_out; /*!< Pointer to data after compression. */
    int len; /*!< Length of output data. */
};

typedef struct ExpandedData {
    int16_t *dataExpanded;
    int iqWidth;
    int numBlocks;
    int numDataElements;
} ExpandedData;

typedef struct CompressedData {
    uint8_t *dataCompressed;
    int iqWidth;
    int numBlocks;
    int numDataElements;
} CompressedData;

static inline __m512i maskUpperWord(const __m512i inData)
{
    const __m512i k_upperWordMask = _mm512_set_epi64(0x0000FFFF0000FFFF, 0x0000FFFF0000FFFF,
                                                     0x0000FFFF0000FFFF, 0x0000FFFF0000FFFF,
                                                     0x0000FFFF0000FFFF, 0x0000FFFF0000FFFF,
                                                     0x0000FFFF0000FFFF, 0x0000FFFF0000FFFF);
    return _mm512_and_epi64(inData, k_upperWordMask);
}

static inline __m512i expLzCnt(const __m512i maxAbs, const __m512i totShiftBits)
{
    const __m512i lzCount = _mm512_lzcnt_epi32(maxAbs);
    return _mm512_subs_epu16(totShiftBits, lzCount);
}

static inline __m512i networkBytePack12b_snc(const __m512i compData)
{
    /// Logical shift left to align network order byte parts
    const __m512i k_shiftLeft = _mm512_set_epi64(0x0000000400000004, 0x0000000400000004,
                                                                   0x0000000400000004, 0x0000000400000004,
                                                                     0x0000000400000004, 0x0000000400000004,
                                                                   0x0000000400000004, 0x0000000400000004);
  const __m512i compDataPacked = _mm512_sllv_epi16(compData, k_shiftLeft);

    /// First epi8 shuffle of even indexed samples
    const __m512i k_byteShuffleMask1 = _mm512_set_epi64(0x0000000000000000, 0x0000000000000000,
                                                                                0x003C3D0038390034, 0x35003031002C2D00,
                                                                                0x2829002425002021, 0x001C1D0018190014,
                                                                                0x15001011000C0D00, 0x0809000405000001);
     uint64_t k_byteMask1 = 0x00006DB6DB6DB6DB;
    //const __m512i compDataShuff1 = _mm512_maskz_permutexvar_epi8(k_byteMask1, k_byteShuffleMask1, compDataPacked);
    const __m512i compDataShuff1 = _mm512_set_epi64(0x0000000400000004, 0x0000000400000004,
                                                                     0x0000000400000004, 0x0000000400000004,
                                                                     0x0000000400000004, 0x0000000400000004,
                                                                     0x0000000400000004, 0x0000000400000004);

    /// Second epi8 shuffle of odd indexed samples
    const __m512i k_byteShuffleMask2 = _mm512_set_epi64(0x0000000000000000, 0x0000000000000000,
                                                                                0x3E3F003A3B003637, 0x003233002E2F002A,
                                                                                0x2B00262700222300, 0x1E1F001A1B001617,
                                                                                0x001213000E0F000A, 0x0B00060700020300);
     uint64_t k_byteMask2 = 0x0000DB6DB6DB6DB6;
    //const __m512i compDataShuff2 = _mm512_maskz_permutexvar_epi8(k_byteMask2, k_byteShuffleMask2, compDataPacked);
    const __m512i compDataShuff2 = _mm512_set_epi64(0x0000000400000004, 0x0000000400000004,
                                                                     0x0000000400000004, 0x0000000400000004,
                                                                     0x0000000400000004, 0x0000000400000004,
                                                                     0x0000000400000004, 0x0000000400000004);

    /// Ternary blend of the two shuffled results
    const __m512i k_ternLogSelect = _mm512_set_epi64(0x0000000000000000, 0x0000000000000000,
                                                                         0xFF0F00FF0F00FF0F, 0x00FF0F00FF0F00FF,
                                                                         0x0F00FF0F00FF0F00, 0xFF0F00FF0F00FF0F,
                                                                         0x00FF0F00FF0F00FF, 0x0F00FF0F00FF0F00);
    return _mm512_ternarylogic_epi64(compDataShuff1, compDataShuff2, k_ternLogSelect, 0xd8);
}

static inline __m512i networkBytePack9b_snc(const __m512i compData)
 {
   /// Logical shift left to align network order byte parts
    const __m512i k_shiftLeft = _mm512_set_epi64(0x0000000100020003, 0x0004000500060007,
                                                 0x0000000100020003, 0x0004000500060007,
                                                 0x0000000100020003, 0x0004000500060007,
                                                 0x0000000100020003, 0x0004000500060007);
    const __m512i compDataPacked = _mm512_sllv_epi16(compData, k_shiftLeft);

    /// First epi8 permute of even indexed samples
    const __m512i k_byteShuffleMask1 = _mm512_set_epi64(0x0000000000000000, 0x0000000000000000,
                                                        0x0000000000000000, 0x00000000003C3D38,
                                                        0x3934353031002C2D, 0x282924252021001C,
                                                        0x1D18191415101100, 0x0C0D080904050001);
     uint64_t k_byteMask1 = 0x00000007FBFDFEFF;
    //const __m512i compDataShuff1 = _mm512_maskz_permutexvar_epi8(k_byteMask1, k_byteShuffleMask1, compDataPacked);
    const __m512i compDataShuff1 = _mm512_set_epi64(0x0000000400000004, 0x0000000400000004,
                                                                     0x0000000400000004, 0x0000000400000004,
                                                                     0x0000000400000004, 0x0000000400000004,
                                                                     0x0000000400000004, 0x0000000400000004);

    /// Second epi8 permute of odd indexed samples
    const __m512i k_byteShuffleMask2 = _mm512_set_epi64(0x0000000000000000, 0x0000000000000000,
                                                        0x0000000000000000, 0x000000003E3F3A3B,
                                                        0x36373233002E2F2A, 0x2B26272223001E1F,
                                                        0x1A1B16171213000E, 0x0F0A0B0607020300);
     uint64_t k_byteMask2 = 0x0000000FF7FBFDFE;
    //__m512i compDataShuff2 = _mm512_maskz_permutexvar_epi8(k_byteMask2, k_byteShuffleMask2, compDataPacked);
    const __m512i compDataShuff2 = _mm512_set_epi64(0x0000000400000004, 0x0000000400000004,
                                                                     0x0000000400000004, 0x0000000400000004,
                                                                     0x0000000400000004, 0x0000000400000004,
                                                                     0x0000000400000004, 0x0000000400000004);

    /// Ternary blend of the two shuffled results
    const __m512i k_ternLogSelect = _mm512_set_epi64(0x0000000000000000, 0x0000000000000000,
                                                     0x0000000000000000, 0x00000000FF01FC07,
                                                     0xF01FC07F00FF01FC, 0x07F01FC07F00FF01,
                                                     0xFC07F01FC07F00FF, 0x01FC07F01FC07F00);
    return _mm512_ternarylogic_epi64(compDataShuff1, compDataShuff2, k_ternLogSelect, 0xd8);
  }

uint8_t computeExponent_1RB(const ExpandedData * dataIn,const __m512i totShiftBits)
{
    // const __m512i* rawData = (const __m512i*)(dataIn->dataExpanded);
    const __m512i rawData  = _mm512_loadu_si512(dataIn->dataExpanded);
    //Abs
    const __m512i rawDataAbs = _mm512_abs_epi16(rawData);

    // No need to do a full horizontal max operation here, just do a max IQ step,
    // compute the exponents and then use a reduce max over all exponent values.
    // This is the fastest way to handle a single RB.
    const __m512i rawAbsIQSwap = _mm512_rol_epi32(rawDataAbs,16);
    const __m512i maxAbsIQ = _mm512_max_epi16(rawDataAbs, rawAbsIQSwap);

    // Calculate exponent
    const __m512i maxAbsIQ32 = maskUpperWord(maxAbsIQ);
    const __m512i exps = expLzCnt(maxAbsIQ32, totShiftBits);

    // At this point we have exponent values for the maximum of each IQ pair.
    // Run a reduce max step to compute the maximum exponent value in the first
    // three lanes - this will give the desired exponent for this RB.
    uint16_t k_expMsk = 0x0FFF;
    return (uint8_t)_mm512_mask_reduce_max_epi32(k_expMsk, exps);
}

void applyCompressionN_1RB_snc(const ExpandedData* dataIn,CompressedData* dataOut,
                        const int numREOffset, const uint8_t thisExp, const int thisRBExpAddr, const uint64_t rbWriteMask, int bfpcExpOffset)
{
    /// Get AVX512 pointer aligned to desired RB
    const __m512i rawDataIn = _mm512_loadu_si512(dataIn->dataExpanded + numREOffset);
    /// Apply the exponent shift
    const __m512i compData = _mm512_srai_epi16(rawDataIn, thisExp);
    /// Pack compressed data network byte order
    __m512i compDataBytePacked_temp;
    if(dataIn->iqWidth == BFPC_12BIT)
        compDataBytePacked_temp = networkBytePack12b_snc(compData);
    else if(dataIn->iqWidth == BFPC_9BIT)
        compDataBytePacked_temp = networkBytePack9b_snc(compData);

    const __m512i compDataBytePacked = compDataBytePacked_temp;
    /// Store exponent first
    dataOut->dataCompressed[thisRBExpAddr] = thisExp + bfpcExpOffset;

    // Now have 1 RB worth of bytes separated into 3 chunks (1 per lane)
    // Use three offset stores to join
    _mm_mask_storeu_epi8(dataOut->dataCompressed + thisRBExpAddr + 1, rbWriteMask, _mm512_extracti64x2_epi64(compDataBytePacked, 0));
    _mm_mask_storeu_epi8(dataOut->dataCompressed + thisRBExpAddr + 1 + dataIn->iqWidth, rbWriteMask, _mm512_extracti64x2_epi64(compDataBytePacked, 1));
    _mm_mask_storeu_epi8(dataOut->dataCompressed + thisRBExpAddr + 1 + (2 * dataIn->iqWidth), rbWriteMask, _mm512_extracti64x2_epi64(compDataBytePacked, 2));

}

int32_t xranlib_compress_snc(const struct xranlib_compress_request *request, struct xranlib_compress_response *response)
{
    struct ExpandedData expandedDataInput;
    struct CompressedData compressedDataOut;
    uint16_t totalRBs = request->numRBs;
    uint16_t remRBs = totalRBs;
    uint16_t len = 0;
    uint16_t block_idx_bytes = 0;
    uint64_t rbWriteMask;
    int bfpcExpOffset = request->bfpc_exp_offset;
    int arg_to_set1epi32;
    int out_width;
//    _mm_prefetch((char*)&(response->data_out[len]),_MM_HINT_T0);
    if(request->iqWidth == BFPC_12BIT)
    {
        arg_to_set1epi32 = 21;
        rbWriteMask = 0x0000000FFFFFFFFF;
    }
    else if(request->iqWidth == BFPC_9BIT)
    {
        arg_to_set1epi32 = 24;
        rbWriteMask = 0x0000000007FFFFFF;
    }
    const __m512i totShiftBits = _mm512_set1_epi32(arg_to_set1epi32);

    expandedDataInput.iqWidth = request->iqWidth;
    expandedDataInput.numDataElements = request->numDataElements;
    out_width = (3 * expandedDataInput.iqWidth) + 1;
    while (remRBs) {
        expandedDataInput.dataExpanded = (int16_t *) &(request->data_in[block_idx_bytes]);
        compressedDataOut.dataCompressed = (int8_t *) &(response->data_out[len]);
        if (remRBs >= 1) {
            expandedDataInput.numBlocks = 1;
  //          _mm_prefetch((char*)&(request->data_in[block_idx_bytes + expandedDataInput.numDataElements]), _MM_HINT_T0);
            uint8_t thisExponent = computeExponent_1RB(&expandedDataInput, totShiftBits);
            applyCompressionN_1RB_snc(&expandedDataInput, &compressedDataOut, 0, thisExponent, 0, rbWriteMask, bfpcExpOffset);
            len += out_width;
    //        _mm_prefetch((char*)&(response->data_out[len]),_MM_HINT_T0);
            block_idx_bytes += 1*expandedDataInput.numDataElements;
            remRBs = remRBs - 1;
        }
    }

    //Set the response length
    return response->len = ((3 * expandedDataInput.iqWidth) + 1) * totalRBs;
}

static inline uint64_t rdtsc_b()
{
    union{
        uint64_t tsc;
        __extension__ struct {
            uint32_t lo;
            uint32_t hi;
        };
    }tsc;
    /*
    //asm volatile ("cpuid\t\n");
    asm volatile ("rdtsc;"
    "shl $32,%%rdx;"
    "or %%rdx,%%rax;": "=a" (tsc.tsc) :: "%rcx", "%rdx", "memory");
    */
    asm volatile ("CPUID\n\t"
            "RDTSC\n\t"
            "mov %%edx, %0\n\t"
            "mov %%eax, %1\n\t": "=r" (tsc.hi), "=r" (tsc.lo)::
            "%rax", "%rbx", "%rcx", "%rdx");
    return tsc.tsc;
}

static inline uint64_t rdtsc_e()
{
    union{
        uint64_t tsc;
        __extension__ struct {
            uint32_t lo;
            uint32_t hi;
        };
    }tsc;
    /*
       asm volatile ("rdtscp;"
       "shl $32,%%rdx;"
       "or %%rdx,%%rax;"
       : "=a" (tsc.tsc) :: "%rcx", "%rdx" );
       asm volatile ("cpuid");
       */
    asm volatile("RDTSCP\n\t"
            "mov %%edx, %0\n\t"
            "mov %%eax, %1\n\t"
            "CPUID\n\t": "=r" (tsc.hi), "=r" (tsc.lo)::
            "%rax", "%rbx", "%rcx", "%rdx");

    return tsc.tsc;
}
#ifdef ENABLE_PCM
void* pcm_loop( void* para)
{
    int32_t lcore_id;
    int32_t ret;
    cpu_set_t cpuset;
    pthread_t thread_id;

    thread_id = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(10, &cpuset);
    ret = pthread_setaffinity_np(thread_id, sizeof(cpuset), &cpuset);

#define PIPE_BUF_SIZE 2
    char buf[PIPE_BUF_SIZE];
    lcore_id = pcm_getcpu();
    printf("pcm is running on core: %d\n", lcore_id);

    /* 
    * Initialization PCM structure
    */
#ifdef PCM_DYNAMIC_LIB
	void * handle = dlopen("libpcm.so", RTLD_LAZY);
	if(!handle) {
		printf("Abort: could not (dynamically) load shared library \n");
		exit(-1);
	}

	PCM.pcm_c_build_core_event = (int (*)(uint8_t, const char *)) dlsym(handle, "pcm_c_build_core_event");
	PCM.pcm_c_init = (int (*)()) dlsym(handle, "pcm_c_init");
	PCM.pcm_c_start = (void (*)()) dlsym(handle, "pcm_c_start");
	PCM.pcm_c_stop = (void (*)()) dlsym(handle, "pcm_c_stop");
	PCM.pcm_c_get_cycles = (uint64_t (*)(uint32_t)) dlsym(handle, "pcm_c_get_cycles");
	PCM.pcm_c_get_instr = (uint64_t (*)(uint32_t)) dlsym(handle, "pcm_c_get_instr");
	PCM.pcm_c_get_core_event = (uint64_t (*)(uint32_t,uint32_t)) dlsym(handle, "pcm_c_get_core_event");
#else
	PCM.pcm_c_build_core_event = pcm_c_build_core_event;
	PCM.pcm_c_init = pcm_c_init;
	PCM.pcm_c_start = pcm_c_start;
	PCM.pcm_c_stop = pcm_c_stop;
	PCM.pcm_c_get_cycles = pcm_c_get_cycles;
	PCM.pcm_c_get_instr = pcm_c_get_instr;
	PCM.pcm_c_get_core_event = pcm_c_get_core_event;
#endif

    /*
    * Initialize PCM facilities
    */
    if(PCM.pcm_c_init == NULL || PCM.pcm_c_start == NULL || PCM.pcm_c_stop == NULL ||
            PCM.pcm_c_get_cycles == NULL || PCM.pcm_c_get_instr == NULL ||
            PCM.pcm_c_build_core_event == NULL)
        exit(-1);
    // example event, use a configuration table later.
    PCM.pcm_c_build_core_event(0, "umask=0x10,event=0x60,name=OFFCORE_REQUESTS_OUTSTANDING.L3_MISS_DEMAND_DATA_RD,offcore_rsp=0x00");
    //PCM.pcm_c_build_core_event(1, "umask=0x01,event=0xB7,name=OCR.READS_TO_CORE.L3_MISS_LOCAL,offcore_rsp=0x3F0CC00477");
    //PCM.pcm_c_build_core_event(1, "umask=0x20,event=0xd1,name=MEM_LOAD_RETIRED.L3_MISS,offcore_rsp=0x00");
    PCM.pcm_c_build_core_event(1, "umask=0x10,event=0xd1,name=MEM_LOAD_RETIRED.L1_MISS,offcore_rsp=0x00");
    
    printf("PCM Initializating ...\n");
    PCM.pcm_c_init();
    printf("PCM initialization completed.\n");

    buf[0] = 0xff;
    buf[1] = 0xff;

    while(1){
        ret = read(pipefd[READ_PIPE_FD], buf, PIPE_BUF_SIZE);
        if(ret == -1){
            printf("Pipe read error. Exit.\n");
            exit(-1);
        }
        if (buf[0] == 1){
            printf("start pcm.\n");
            PCM.pcm_c_start();
        } 
        else if (buf[0] == 0) {
            lcore_id = buf[1];

            if (lcore_id < 0 || lcore_id > 20){
                printf("invalid core id: %d received. pcm stop command ignored.", lcore_id);
            }
            printf("stop pcm, and generate the measurements.\n");
            PCM.pcm_c_stop();

            printf("C:%lu I:%lu, IPC:%3.2f\n",
                    PCM.pcm_c_get_cycles(lcore_id),
                    PCM.pcm_c_get_instr(lcore_id),
                    (double)PCM.pcm_c_get_instr(lcore_id)/PCM.pcm_c_get_cycles(lcore_id));
            printf("CPU%d E0: %lu\n", lcore_id, PCM.pcm_c_get_core_event(lcore_id,0));
            printf("CPU%d E1: %lu\n", lcore_id, PCM.pcm_c_get_core_event(lcore_id,1));
        }
    }
}
#endif

#define SZ_BLOCK 4096
#define NB_BLOCK 40960
int16_t __attribute__((aligned(64))) ibuf[SZ_BLOCK*NB_BLOCK];
int8_t obuf[SZ_BLOCK*NB_BLOCK];

int warmup(int warmup_cache)
{
    uint64_t start, end;

    if (warmup_cache & 0x1) {
        printf("warm up (initialize) input buffer.\n");
        start = rdtsc_b();
        for ( int i = 0; i < SZ_BLOCK*NB_BLOCK; i++){
            ibuf[i] = ((i % 255) * 133 /7) % 65536; // simple magic to randomize the content 
        }
        end = rdtsc_e();
        printf("Cost of input buffer warmup: %ld\n", end - start);
    } 
    if ( warmup_cache & 0x2){
        printf("warm up (initialize) output buffer.\n");
        start = rdtsc_b();
        bzero(obuf, SZ_BLOCK*NB_BLOCK*sizeof(int8_t));
        end = rdtsc_e();
        printf("Cost of output buffer warmup: %ld\n", end - start);
    } 

    return 0;
}

void main(int argc, char *argv[])
{
	struct xranlib_compress_request request;
    struct xranlib_compress_response response;
    uint64_t start, end, max, min, avg, total, delta;
#define NB_TS 4
    uint64_t timestamp[NB_TS];
    int32_t ts_inx = 0;
    int32_t ret = 0;
    int16_t prefetch_distance = 0;    // distance to prefetch
    int16_t warmup_cache = 0;         // warm up by load/initialize
    int16_t priority = 0;             // set thread priority
    int16_t quiet = 0;                // if print the outliers during runtime
    int16_t logfile = 0;              // if log the statistic to file
    int16_t suspend = 0;              // suspend the execution, allow perf to 
                                      // collect statistic per stop
	int option;

    while ((option = getopt(argc, argv, "d:w:p:qls")) != -1){
        switch (option){
            case 'd':
                prefetch_distance = optarg!=NULL?atoi(optarg):0;
                break;
            case 'w':
              //  printf("option warmup cache specified: %s\n", optarg);
                warmup_cache = atoi(optarg);
                break;
            case 'p':
                priority = atoi(optarg);
                break;
            case 'q':
                quiet = 1;
                break;
            case 'l':
                logfile = 1;
                break;
            case 's':
                suspend = 1;
                break;
            case 'h':
            case '?':
                printf("compress2 [-d <prefetch_distance>] [-w <buffers>]\n" \
                   "\t -d <prefetch_distance> prefetch_distance specifies the distance of the block to be prefetched.\n" \
                   "\t -w <buffers> 1, only warm up the input buffer.\n" \
                   "\t\t 2, only warm up the output buffer.\n" \
                   "\t\t 3, warm up both input and output buffer.\n" \
                   "\t -q quiet, don't print the outlier during first 4096 runs.\n" \
                   "\t -l log the statistic to file.\n" \
                   "\t -s suspend the execution at fixed point, allow perf to collect infomation per stop.\n");
                exit(0);
            break;

        }
    }

    if (priority > 0){
        struct sched_param param;
        int policy = SCHED_FIFO;
        param.sched_priority = 96;

        ret = pthread_setschedparam(pthread_self(), policy, &param);
        if ( ret != 0){
        
            printf("failed to set sched policy :%d and priority 96\n", policy);
        }
    }

    mlockall(MCL_FUTURE);

#ifdef ENABLE_PCM
    int32_t lcore_id;
    char pipebuf[2];
    pthread_t thread_id;
#endif
#define NB_RUNS 1024*1000
    uint16_t zcycles[NB_RUNS];
#define LEN_PER_WRITE 256
    char string[LEN_PER_WRITE];

    max = 0;
    total = 0;
    min = 9999999;
    delta = 0;
    
    if (suspend == 1){
        timestamp[ts_inx++] = rdtsc_b();
        printf("TSC: %ld - Press any key to continue the initialization.\n", timestamp[ts_inx - 1]);
        ret = getchar();
    //    asm volatile ("" : : : "memory");
    }
    
    warmup(warmup_cache);
            
#define SZ_CACHELINE 64
    if (prefetch_distance == -1){
        printf("Prefetching - trying to load all the buffer to cache.\n");
        for ( int i = 0; i < sizeof(int16_t)*SZ_BLOCK*NB_BLOCK/SZ_CACHELINE; i++){
            __builtin_prefetch((char*)((char*)ibuf + i*SZ_CACHELINE), 0, 0);
        }
    }

    request.numRBs = 162;
    request.numDataElements = 24;
    request.iqWidth = BFPC_12BIT;
    request.len = 162 * 16;
    request.bfpc_exp_offset = 0;

#ifdef ENABLE_PCM
    lcore_id = pcm_getcpu();
    ret = pipe(pipefd);
    if (ret == -1){
        printf("Pipe creation failed. exit.\n");
        exit(-1);
    }
        
    ret = pthread_create(&thread_id, NULL, pcm_loop, NULL);
    if (ret == -1){
        printf("pcm thread creation failed. \n");
    }
    pipebuf[0] = 1;
    pipebuf[1] = lcore_id;
	ret = write(pipefd[WRITE_PIPE_FD],pipebuf,2);
    if (ret == -1){
        printf("Failed to send start pcm command.\n");
    }

    printf("sleep 1 second for pcm to start.\n");
    sleep(1);
#endif

    if (suspend == 1){
        timestamp[ts_inx++] = rdtsc_b();
        printf("TSC: %ld - Initialization completed. Press any key to continue.\n", timestamp[ts_inx-1]);
        printf("TSC elasped since last event: %ld\n", timestamp[ts_inx -1] - timestamp[ts_inx - 2]);
        ret = getchar();
    //    asm volatile ("" : : : "memory");
    }
    printf("start compression. \n");
	for(int i = 0; i < NB_RUNS; i++) {
		request.data_in = ibuf+(i%NB_BLOCK)*SZ_BLOCK;
    	response.data_out = (char*)(obuf+(i%NB_BLOCK)*SZ_BLOCK);
#if 0
        if ( prefetch_distance > 0 && i < NB_BLOCK - prefetch_distance ){
            uint16_t * ibufnext = ibuf + (i+prefetch_distance)*SZ_BLOCK;
            char* obufnext = obuf + (i+prefetch_distance)*SZ_BLOCK;
            for (int j = 0; j < NB_BLOCK/SZ_CACHELINE; j++){
                 _mm_prefetch((char*)(ibufnext + j*SZ_CACHELINE), _MM_HINT_T0);
                 _mm_prefetch((char*)(ibufnext + j*SZ_CACHELINE + SZ_CACHELINE/2), _MM_HINT_T0);
                 _mm_prefetch((char*)(obufnext + j*SZ_CACHELINE), _MM_HINT_T0);
            }
        }
#endif

        start = rdtsc_b();
		ret += xranlib_compress_snc(&request, &response);
        end = rdtsc_e();

        delta = end - start;
        zcycles[i] = delta;

        total += delta;

        if (max < delta) max = delta;
        if (min > delta) min = delta;
#if 0
    	if(delta > 12000 && ( quiet == 0 || i >= NB_BLOCK)) // threshold to highlight the outliers
			printf("%-8d: %ld cycles, request: %p, response: %p, obuf: %p\n", i, delta, 
            request, response, obuf+(i%NB_BLOCK)*SZ_BLOCK);
#endif
        if ( suspend == 1 && i == NB_BLOCK) {
            timestamp[ts_inx++] = rdtsc_b();
            printf("TSC: %ld - First round completed. Press any key to continue.\n", timestamp[ts_inx - 1]);
            printf("TSC elasped since last event: %ld\n", timestamp[ts_inx -1] - timestamp[ts_inx - 2]);
            ret = getchar();
      //      asm volatile ("" : : : "memory");
        }
#ifdef ENABLE_PCM
        if ( i == NB_BLOCK) {
            pipebuf[0] = 0;
            ret = write(pipefd[WRITE_PIPE_FD], pipebuf, 2);
            if (ret == -1){
                printf("Failed to send stop pcm command. \n");
            }
            sleep(2);
            pipebuf[0] = 1;
            ret = write(pipefd[WRITE_PIPE_FD], pipebuf, 2);
            if (ret == -1){
                printf("Failed to send start pcm command. \n");
            }
            sleep(1);
        }
#endif
	}
#ifdef ENABLE_PCM
    pipebuf[0] = 0;
    ret = write(pipefd[WRITE_PIPE_FD], pipebuf, 2);
    if (ret == -1){
        printf("Failed to send stop pcm command. \n");
    }
#endif
    avg = total/(NB_RUNS);
	
	printf("total time(cycles) = %ld, ret = %ld\nmax = %ld, avg = %ld, min = %ld\n", total, ret, max, avg, min);
    if (suspend == 1){
        timestamp[ts_inx] = rdtsc_b();
        printf("TSC: %ld - completed.\n", timestamp[ts_inx]);
        printf("TSC elasped since last event: %ld\n", timestamp[ts_inx] - timestamp[ts_inx - 1]);
    }
        
    if (logfile == 1){
        FILE *fp;
        char filename[256];
        snprintf(filename, 256, "timing_compress2-w%d-pd%d.txt", warmup_cache, prefetch_distance);
        fp = fopen(filename, "w");
        if (fp == NULL){
            printf("failed to open file to log the detailed timing.\n");
        }else{
            snprintf(string, LEN_PER_WRITE, "Occurence\tcycles\n");
            fprintf(fp, "%s", string);
            for (int i = 0; i < NB_RUNS; i++){
                snprintf(string, LEN_PER_WRITE, "%9d\t%6d\n", i, zcycles[i]);
                fprintf(fp, "%s", string);
            }
            fclose(fp);
        }
    }
#ifdef ENABLE_PCM
    printf("PCM thread is still running, press CTRL-C to exit.\n");
    void* tret;
    pthread_join(thread_id, &tret);
#endif
}
