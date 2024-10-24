#include <map>
#include "/home/skw404/MNN_SFC/MNN/source/backend/cpu/x86_x64/avx/Vec8.hpp"
#include "/home/skw404/MNN_SFC/MNN/source/backend/cpu/x86_x64/avx/FunctionSummary.hpp"
#include "/home/skw404/MNN_SFC/MNN/source/core/MemoryFormater.h"
#include <iostream>
#include "/home/skw404/MNN_SFC/MNN/source/backend/cpu/compute/ConvolutionPackWinograd.hpp"
#include <math.h>
#include "/home/skw404/MNN_SFC/MNN/source/backend/cpu/compute/CommonOptFunction.h"
#include "/home/skw404/MNN_SFC/MNN/source/core/Concurrency.h"
#include "/home/skw404/MNN_SFC/MNN/source/backend/cpu/compute/ConvOpt.h"
#include "/home/skw404/MNN_SFC/MNN/source/core/Macro.h"
#include "/home/skw404/MNN_SFC/MNN/source/core/TensorUtils.hpp"
#include "/home/skw404/MNN_SFC/MNN/source/math/WingoradGenerater.hpp"
#include <MNN/AutoTime.hpp>
#include "/home/skw404/MNN_SFC/MNN/source/core/MemoryFormater.h"
#include <iostream>

#define PACK_UNIT 8
using VecType = Vec8;
using namespace MNN::Math;

auto core = static_cast<CPUBackend*>(backend())->functions();


const float src[8*8]={1,2,3,4,
                    5,6,7,8,
                    9,10,11,12,
                    13,14,15,16};

float dst_wino[8*8]={0};
float dst_sfc[8*8]={0};


void _sourceUnrollTransformUnit8x8(const float* srcBlock, float* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    constexpr size_t srcUnit = 8; // srcUnit
    Vec8 buf0 = Vec8::load(srcBlock + 0 * srcStep);
    Vec8 buf1 = Vec8::load(srcBlock + 1 * srcStep);
    Vec8 buf2 = Vec8::load(srcBlock + 2 * srcStep);
    Vec8 buf3 = Vec8::load(srcBlock + 3 * srcStep);
    Vec8 buf4 = Vec8::load(srcBlock + 4 * srcStep);
    Vec8 buf5 = Vec8::load(srcBlock + 5 * srcStep);
    Vec8 buf6 = Vec8::load(srcBlock + 6 * srcStep);
    Vec8 buf7 = Vec8::load(srcBlock + 7 * srcStep);
// #pragma unroll(srcUnit - 1)
    for (int i = 0; i < srcUnit - 1; ++i) { //Nw iteration
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        Vec8 mid0, mid1, mid2;
        mid0     = Vec8::fma(Vec8::fma(buf6, buf2, Vec8(36)), buf4, Vec8(-13));
        mid1     = Vec8::fma(Vec8::fma(buf4, buf0, Vec8(36)), buf2, Vec8(-13));
        Vec8 m0 = mid1 - mid0;

        mid2     = Vec8::fma(Vec8::fma(buf5, buf1, Vec8(36)), buf3, Vec8(-13));
        Vec8 m1 = mid0 + mid2;
        Vec8 m2 = mid0 - mid2;
        mid1     = Vec8::fma(Vec8::fma(buf7, buf3, Vec8(36)), buf5, Vec8(-13));
        Vec8 m7 = mid1 - mid2;

        mid0     = Vec8::fma(Vec8::fma(buf6, buf2, Vec8(9)), buf4, Vec8(-10));
        mid1     = Vec8::fma(buf5, buf1, Vec8(18)) + Vec8::fma(buf5, buf3, Vec8(-20));
        mid2     = Vec8::fma(buf5 * 3, buf1, Vec8(12));
        Vec8 m3 = mid0 + mid1;
        Vec8 m4 = mid0 - mid1;

        mid0     = Vec8::fma(Vec8::fma(buf6, buf2, Vec8(4)), buf4, Vec8(-5));
        mid1     = Vec8::fma(mid2, buf3, Vec8(-15));
        Vec8 m5 = mid0 + mid1;
        Vec8 m6 = mid0 - mid1;

        buf0 = Vec8::load(srcFloatPtr + 0 * srcStep);
        Vec8::save(dstFloatPtr + 0 * dstStep, m0);
        buf1 = Vec8::load(srcFloatPtr + 1 * srcStep);
        Vec8::save(dstFloatPtr + 1 * dstStep, m1);
        buf2 = Vec8::load(srcFloatPtr + 2 * srcStep);
        Vec8::save(dstFloatPtr + 2 * dstStep, m2);
        buf3 = Vec8::load(srcFloatPtr + 3 * srcStep);
        Vec8::save(dstFloatPtr + 3 * dstStep, m3);
        buf4 = Vec8::load(srcFloatPtr + 4 * srcStep);
        Vec8::save(dstFloatPtr + 4 * dstStep, m4);
        buf5 = Vec8::load(srcFloatPtr + 5 * srcStep);
        Vec8::save(dstFloatPtr + 5 * dstStep, m5);
        buf6 = Vec8::load(srcFloatPtr + 6 * srcStep);
        Vec8::save(dstFloatPtr + 6 * dstStep, m6);
        buf7 = Vec8::load(srcFloatPtr + 7 * srcStep);
        Vec8::save(dstFloatPtr + 7 * dstStep, m7);
    }

    auto dstFloatPtr = (float*)(dstStart + (srcUnit - 1) * dstRowStep);
    Vec8 mid0, mid1, mid2;
    mid0     = Vec8::fma(Vec8::fma(buf6, buf2, Vec8(36)), buf4, Vec8(-13));
    mid1     = Vec8::fma(Vec8::fma(buf4, buf0, Vec8(36)), buf2, Vec8(-13));
    Vec8 m0 = mid1 - mid0;

    mid2     = Vec8::fma(Vec8::fma(buf5, buf1, Vec8(36)), buf3, Vec8(-13));
    Vec8 m1 = mid0 + mid2;
    Vec8 m2 = mid0 - mid2;
    mid1     = Vec8::fma(Vec8::fma(buf7, buf3, Vec8(36)), buf5, Vec8(-13));
    Vec8 m7 = mid1 - mid2;

    mid0     = Vec8::fma(Vec8::fma(buf6, buf2, Vec8(9)), buf4, Vec8(-10));
    mid1     = Vec8::fma(buf5, buf1, Vec8(18)) + Vec8::fma(buf5, buf3, Vec8(-20));
    mid2     = Vec8::fma(buf5 * 3, buf1, Vec8(12));
    Vec8 m3 = mid0 + mid1;
    Vec8 m4 = mid0 - mid1;

    mid0     = Vec8::fma(Vec8::fma(buf6, buf2, Vec8(4)), buf4, Vec8(-5));
    mid1     = Vec8::fma(mid2, buf3, Vec8(-15));
    Vec8 m5 = mid0 + mid1;
    Vec8 m6 = mid0 - mid1;

    Vec8::save(dstFloatPtr + 0 * dstStep, m0);
    Vec8::save(dstFloatPtr + 1 * dstStep, m1);
    Vec8::save(dstFloatPtr + 2 * dstStep, m2);
    Vec8::save(dstFloatPtr + 3 * dstStep, m3);
    Vec8::save(dstFloatPtr + 4 * dstStep, m4);
    Vec8::save(dstFloatPtr + 5 * dstStep, m5);
    Vec8::save(dstFloatPtr + 6 * dstStep, m6);
    Vec8::save(dstFloatPtr + 7 * dstStep, m7);

}

//test_avx2_wino

int main(){

    _sourceUnrollTransformUnit8x8(src,dst_wino,4,4,1,1);
    for(int i=0;i<16;i++)
    {
        std::cout<<dst_wino[i]<<",";
    }

    std::cout<<std::endl;

    return 0;


}
