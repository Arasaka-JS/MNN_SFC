//
//  ConvolutionPackWinograd.cpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionPackWinograd.hpp"
#include <math.h>
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "math/WingoradGenerater.hpp"
#include "math/Matrix.hpp"
#include <MNN/AutoTime.hpp>
#include "core/MemoryFormater.h"
#include <iostream>
#include <fstream>


#include <map>
#include "backend/cpu/x86_x64/avx/Vec8.hpp"
#include "backend/cpu/x86_x64/avx/FunctionSummary.hpp"
#include "core/MemoryFormater.h"
#include <iostream>

#include <chrono>


#define PACK_UNIT 8




constexpr int FULSE_THRESHHOLD_NUMERATOR = 10;
constexpr int FULSE_THRESHHOLD_DENOMINATOR = 10;

using namespace MNN::Math;


using VecType = Vec8;

#define TRANSPOSE_24X8_SAVE()                                 \
    VecType s0  = VecType::load(srcPtr + 0 * packCUnit);      \
    VecType s3  = VecType::load(srcPtr + 1 * packCUnit);      \
    VecType s6  = VecType::load(srcPtr + 2 * packCUnit);      \
    VecType s9  = VecType::load(srcPtr + 3 * packCUnit);      \
    VecType s12 = VecType::load(srcPtr + 4 * packCUnit);      \
    VecType s15 = VecType::load(srcPtr + 5 * packCUnit);      \
    VecType s18 = VecType::load(srcPtr + 6 * packCUnit);      \
    VecType s21 = VecType::load(srcPtr + 7 * packCUnit);      \
                                                              \
    VecType s1  = VecType::load(srcPtr + 8 * packCUnit);      \
    VecType s4  = VecType::load(srcPtr + 9 * packCUnit);      \
    VecType s7  = VecType::load(srcPtr + 10 * packCUnit);     \
    VecType s10 = VecType::load(srcPtr + 11 * packCUnit);     \
    VecType s13 = VecType::load(srcPtr + 12 * packCUnit);     \
    VecType s16 = VecType::load(srcPtr + 13 * packCUnit);     \
    VecType s19 = VecType::load(srcPtr + 14 * packCUnit);     \
    VecType s22 = VecType::load(srcPtr + 15 * packCUnit);     \
                                                              \
    VecType s2  = VecType::load(srcPtr + 16 * packCUnit);     \
    VecType s5  = VecType::load(srcPtr + 17 * packCUnit);     \
    VecType s8  = VecType::load(srcPtr + 18 * packCUnit);     \
    VecType s11 = VecType::load(srcPtr + 19 * packCUnit);     \
    VecType s14 = VecType::load(srcPtr + 20 * packCUnit);     \
    VecType s17 = VecType::load(srcPtr + 21 * packCUnit);     \
    VecType s20 = VecType::load(srcPtr + 22 * packCUnit);     \
    VecType s23 = VecType::load(srcPtr + 23 * packCUnit);     \
                                                              \
    VecType::transpose8(s0, s3, s6, s9, s12, s15, s18, s21);  \
    VecType::transpose8(s1, s4, s7, s10, s13, s16, s19, s22); \
    VecType::transpose8(s2, s5, s8, s11, s14, s17, s20, s23); \
    /* to-optimize: interleave load and save in loop*/        \
    VecType::save(srcPtr + 0 * packCUnit, s0);                \
    VecType::save(srcPtr + 1 * packCUnit, s1);                \
    VecType::save(srcPtr + 2 * packCUnit, s2);                \
    VecType::save(srcPtr + 3 * packCUnit, s3);                \
    VecType::save(srcPtr + 4 * packCUnit, s4);                \
    VecType::save(srcPtr + 5 * packCUnit, s5);                \
    VecType::save(srcPtr + 6 * packCUnit, s6);                \
    VecType::save(srcPtr + 7 * packCUnit, s7);                \
    VecType::save(srcPtr + 8 * packCUnit, s8);                \
    VecType::save(srcPtr + 9 * packCUnit, s9);                \
    VecType::save(srcPtr + 10 * packCUnit, s10);              \
    VecType::save(srcPtr + 11 * packCUnit, s11);              \
    VecType::save(srcPtr + 12 * packCUnit, s12);              \
    VecType::save(srcPtr + 13 * packCUnit, s13);              \
    VecType::save(srcPtr + 14 * packCUnit, s14);              \
    VecType::save(srcPtr + 15 * packCUnit, s15);              \
    VecType::save(srcPtr + 16 * packCUnit, s16);              \
    VecType::save(srcPtr + 17 * packCUnit, s17);              \
    VecType::save(srcPtr + 18 * packCUnit, s18);              \
    VecType::save(srcPtr + 19 * packCUnit, s19);              \
    VecType::save(srcPtr + 20 * packCUnit, s20);              \
    VecType::save(srcPtr + 21 * packCUnit, s21);              \
    VecType::save(srcPtr + 22 * packCUnit, s22);              \
    VecType::save(srcPtr + 23 * packCUnit, s23);


void WeightTransSFC(const MNN::Tensor* dest, const MNN::Tensor* source, bool ciFirst = false)
{
    
    //SFC
    std::shared_ptr<MNN::Tensor> mG, mG_Right;
    mG.reset(Matrix::create(3, 10));
    mG_Right.reset(Matrix::create(10, 3));

    float* mG_ptr = mG->host<float>();
    float* mG_Right_ptr = mG_Right->host<float>();

    // 定义要复制的矩阵 WeightH
    float WeightH[10][3] = {
        { 1, 1, 1},
        { 0, 1, 1},
        {-1,-1, 0},
        {-1, 0, 1},
        {-1, 0, 1},
        { 1,-1, 0},
        { 0,-1, 1},
        { 1,-1, 1},
        { 1, 0, 0},
        { 0, 0, 1}
    };

    // 将 WeightH 矩阵的值逐个赋值给 mG_ptr
    for (int i = 0; i < 10; ++i) 
    {
        for (int j = 0; j < 3; ++j) 
        {
            mG_ptr[i * 3 + j] = WeightH[i][j];
        }
    }
    // 将 WeightH 的转置矩阵赋值给 mG_Right_Ptr
    for (int i = 0; i < 10; ++i) 
    {
        for (int j = 0; j < 3; ++j) 
        {
        // 在转置矩阵中，mG_Right_Ptr[j * 10 + i] 对应 WeightH[i][j]
            mG_Right_ptr[j * 10 + i] = WeightH[i][j];
        }   
    }


    int ci          = source->length(1);
    int co          = source->length(0);
    int unitCi      = dest->length(3);
    int unitCo      = dest->length(4);
    auto alpha      = 8;

    if (ci % unitCi != 0 || co % unitCo != 0) {
        ::memset(dest->host<float>(), 0, dest->size());
    }
    std::shared_ptr<MNN::Tensor> M(MNN::Math::Matrix::create(3, 10));
    std::shared_ptr<MNN::Tensor> K(MNN::Math::Matrix::createShape(3, 3));
    std::shared_ptr<MNN::Tensor> K_Transform(MNN::Math::Matrix::create(10, 10));
    auto weightPtr      = source->host<float>();
    auto KTransformData = K_Transform->host<float>();
    int lCi = unitCo;
    int lCo = 1;
    if (ciFirst) {
        lCi = 1;
        lCo = unitCi;
    }
    for (int oz = 0; oz < co; ++oz) {
        int mKernelY =3;int mKernelX =3;
        auto srcOz = weightPtr + oz * ci * mKernelY * mKernelX;

        int ozC4 = oz / unitCo;
        int mx   = oz % unitCo;

        auto dstOz = dest->host<float>() + dest->stride(1) * ozC4 + mx * lCo;
        for (int sz = 0; sz < ci; ++sz) {
            int szC4         = sz / unitCi;
            int my           = sz % unitCi;
            auto srcSz       = srcOz + mKernelY * mKernelX * sz;
            K->buffer().host = (uint8_t*)srcSz;
            // M = G * K
            MNN::Math::Matrix::multi(M.get(), mG.get(), K.get());
            //MNN_PRINT("M=\n");
            //MNN::Math::Matrix::print(M.get());
            //MNN_PRINT("mG=\n");
            //MNN::Math::Matrix::print(mG.get());
            //MNN_PRINT("mG_Right=\n");
            //MNN::Math::Matrix::print(mG_Right.get());
            // K_Transform = M*GT
            MNN::Math::Matrix::multi(K_Transform.get(), M.get(), mG_Right.get());

            auto dstSz = dstOz + szC4 * dest->stride(2) + my * lCi;

            for (int i = 0; i < 10 * 10; ++i) {
                *(dstSz + i * dest->stride(0)) = KTransformData[i];
            }
        }
    }
}




static void sourceUnrollTransform_sfc(const float* srcBlock, float* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    //extern bool use_sfc;
    //if(use_sfc){std::cout<<"hello"<<std::endl;}
    // for(int i =0;i<8;i++)
    // {
    //     std::cout<<srcBlock[i*srcStep]<<",";
    // }

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

        Vec8 m0 = buf1+buf2+buf3+buf4+buf5+buf6;
        Vec8 m1 = buf1+buf2-buf4-buf5;
        Vec8 m2 = -buf2-buf3+buf5+buf6;
        Vec8 m3 = buf1-buf3-buf4+buf6;
        Vec8 m4 = buf1-buf3+buf4-buf6;
        Vec8 m5 = -buf2+buf3-buf5+buf6;
        Vec8 m6 = buf1-buf2+buf4-buf5;
        Vec8 m7 = buf1-buf2+buf3-buf4+buf5-buf6;
        Vec8 m8 = buf0-buf6;
        Vec8 m9 = -buf1+buf7;

        
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

        Vec8::save(dstFloatPtr + 8 * dstStep, m8);
        Vec8::save(dstFloatPtr + 9 * dstStep, m9);

    }

    auto dstFloatPtr = (float*)(dstStart + (srcUnit - 1) * dstRowStep);
        Vec8 m0 = buf1+buf2+buf3+buf4+buf5+buf6;
        Vec8 m1 = buf1+buf2-buf4-buf5;
        Vec8 m2 = -buf2-buf3+buf5+buf6;
        Vec8 m3 = buf1-buf3-buf4+buf6;
        Vec8 m4 = buf1-buf3+buf4-buf6;
        Vec8 m5 = -buf2+buf3-buf5+buf6;
        Vec8 m6 = buf1-buf2+buf4-buf5;
        Vec8 m7 = buf1-buf2+buf3-buf4+buf5-buf6;
        Vec8 m8 = buf0-buf6;
        Vec8 m9 = -buf1+buf7;
    

    Vec8::save(dstFloatPtr + 0 * dstStep, m0);
    Vec8::save(dstFloatPtr + 1 * dstStep, m1);
    Vec8::save(dstFloatPtr + 2 * dstStep, m2);
    Vec8::save(dstFloatPtr + 3 * dstStep, m3);
    Vec8::save(dstFloatPtr + 4 * dstStep, m4);
    Vec8::save(dstFloatPtr + 5 * dstStep, m5);
    Vec8::save(dstFloatPtr + 6 * dstStep, m6);
    Vec8::save(dstFloatPtr + 7 * dstStep, m7);
    Vec8::save(dstFloatPtr + 8 * dstStep, m8);
    Vec8::save(dstFloatPtr + 9 * dstStep, m9);

}



static void sourceUnrollTransform_wino(const float* srcBlock, float* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    
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

        // std::cout<<"dst data"<<std::endl;
        // for(int j=0;j<8;j++)
        // {
        //     std::cout<<*(dstFloatPtr+j*dstStep)<<",";
        // }
        // std::cout<<std::endl;
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



static void sourceTransformUnitPack24_sfc(float* srcBlock, float* dstStart, size_t dstStep){

    // source transform D * B. register number : (srcUnit + 1) * EPack/packCUnit = 27
    // todo: impliment
    constexpr int Nh = 8; // srcUnit
    constexpr int ePack = 24;
    constexpr size_t packCUnit = 8;
    const size_t loadTransposeStride = packCUnit * ePack;
    float* srcPtr = srcBlock;
    for (int iNh = 0; iNh < Nh; ++iNh)
    {
        // register number : ePack
        TRANSPOSE_24X8_SAVE();
        srcPtr += loadTransposeStride;
    }

    //     MNN_PRINT("winograd in BT*D*B, transpose, loadTransposeStride:%zu, dstStep:%zu\n", loadTransposeStride, dstStep);
    // formatMatrix((const float*)srcBlock, {Nh, static_cast<int>(packCUnit), ePack});

    srcPtr = srcBlock;
    float* dstPtr = dstStart;
    for (int i4c = 0; i4c < packCUnit; ++i4c)
    {
        VecType s00 = VecType::load(srcPtr + 0 * loadTransposeStride + 0 * packCUnit);
        VecType s01 = VecType::load(srcPtr + 0 * loadTransposeStride + 1 * packCUnit);
        VecType s02 = VecType::load(srcPtr + 0 * loadTransposeStride + 2 * packCUnit);

        VecType s10 = VecType::load(srcPtr + 1 * loadTransposeStride + 0 * packCUnit);
        VecType s11 = VecType::load(srcPtr + 1 * loadTransposeStride + 1 * packCUnit);
        VecType s12 = VecType::load(srcPtr + 1 * loadTransposeStride + 2 * packCUnit);

        VecType s20 = VecType::load(srcPtr + 2 * loadTransposeStride + 0 * packCUnit);
        VecType s21 = VecType::load(srcPtr + 2 * loadTransposeStride + 1 * packCUnit);
        VecType s22 = VecType::load(srcPtr + 2 * loadTransposeStride + 2 * packCUnit);

        VecType s30 = VecType::load(srcPtr + 3 * loadTransposeStride + 0 * packCUnit);
        VecType s31 = VecType::load(srcPtr + 3 * loadTransposeStride + 1 * packCUnit);
        VecType s32 = VecType::load(srcPtr + 3 * loadTransposeStride + 2 * packCUnit);

        VecType s40 = VecType::load(srcPtr + 4 * loadTransposeStride + 0 * packCUnit);
        VecType s41 = VecType::load(srcPtr + 4 * loadTransposeStride + 1 * packCUnit);
        VecType s42 = VecType::load(srcPtr + 4 * loadTransposeStride + 2 * packCUnit);

        VecType s50 = VecType::load(srcPtr + 5 * loadTransposeStride + 0 * packCUnit);
        VecType s51 = VecType::load(srcPtr + 5 * loadTransposeStride + 1 * packCUnit);
        VecType s52 = VecType::load(srcPtr + 5 * loadTransposeStride + 2 * packCUnit);

        VecType s60 = VecType::load(srcPtr + 6 * loadTransposeStride + 0 * packCUnit);
        VecType s61 = VecType::load(srcPtr + 6 * loadTransposeStride + 1 * packCUnit);
        VecType s62 = VecType::load(srcPtr + 6 * loadTransposeStride + 2 * packCUnit);

        VecType s70 = VecType::load(srcPtr + 7 * loadTransposeStride + 0 * packCUnit);
        VecType s71 = VecType::load(srcPtr + 7 * loadTransposeStride + 1 * packCUnit);
        VecType s72 = VecType::load(srcPtr + 7 * loadTransposeStride + 2 * packCUnit);


        // to-try: reorder complicated commpute of 8x8
        auto ep0 = s10+s20+s30+s40+s50+s60;
        auto ep1 = s11+s21+s31+s41+s51+s61;
        auto ep2 = s12+s22+s32+s42+s52+s62;
        VecType::save(dstPtr + 0 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 0 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 0 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10+s20-s40-s50;
        ep1 = s11+s21-s41-s51;
        ep2 = s12+s22-s42-s52;
        VecType::save(dstPtr + 1 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 1 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 1 * dstStep + 2 * packCUnit, ep2);

        ep0 = -s20-s30+s50+s60;
        ep1 = -s21-s31+s51+s61;
        ep2 = -s22-s32+s52+s62;
        VecType::save(dstPtr + 2 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 2 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 2 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10-s30-s40+s60;
        ep1 = s11-s31-s41+s61;
        ep2 = s12-s32-s42+s62;
        VecType::save(dstPtr + 3 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 3 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 3 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10-s30+s40-s60;
        ep1 = s11-s31+s41-s61;
        ep2 = s12-s32+s42-s62;
        VecType::save(dstPtr + 4 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 4 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 4 * dstStep + 2 * packCUnit, ep2);

        ep0 = -s20+s30-s50+s60;
        ep1 = -s21+s31-s51+s61;
        ep2 = -s22+s32-s52+s62;
        VecType::save(dstPtr + 5 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 5 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 5 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10-s20+s40-s50;
        ep1 = s11-s21+s41-s51;
        ep2 = s12-s22+s42-s52;
        VecType::save(dstPtr + 6 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 6 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 6 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10-s20+s30-s40+s50-s60;
        ep1 = s11-s21+s31-s41+s51-s61;
        ep2 = s12-s22+s32-s42+s52-s62;
        VecType::save(dstPtr + 7 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 7 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 7 * dstStep + 2 * packCUnit, ep2);

        ep0 = s00-s60;
        ep1 = s01-s61;
        ep2 = s02-s62;
        VecType::save(dstPtr + 8 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 8 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 8 * dstStep + 2 * packCUnit, ep2);

        ep0 = -s10+s70;
        ep1 = -s11+s71;
        ep2 = -s12+s72;
        VecType::save(dstPtr + 9 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 9 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 9 * dstStep + 2 * packCUnit, ep2);

        srcPtr += ePack;
        dstPtr += ePack;
    }
}



static void sourceTransformUnitPack24_wino(float* srcBlock, float* dstStart, size_t dstStep) {

    // source transform D * B. register number : (srcUnit + 1) * EPack/packCUnit = 27
    // todo: impliment
    constexpr int Nh = 8; // srcUnit
    constexpr int ePack = 24;
    constexpr size_t packCUnit = 8;
    // constexpr size_t packCUnit = 1;
    const size_t loadTransposeStride = packCUnit * ePack;
    float* srcPtr = srcBlock;


    // //TRANSPOSE_24X8_SAVE() TEST
    // std::cout<<"TRANSPOSE_24X8_SAVE() TEST"<<std::endl;
    // float a[24*8];
    // for(int i=0;i<24*8;i++)
    // {a[i]=i;}
    
    // srcPtr = a ;
    // for(int i=0;i<24*8;i++)
    // {
    //     std::cout<<*(srcPtr+i)<<",";
    // }
    // std::cout<<std::endl;
    // TRANSPOSE_24X8_SAVE();
    // for(int i=0;i<24*8;i++)
    // {
    //     std::cout<<*(srcPtr+i)<<",";
    // }
    // std::cout<<std::endl;
    // return;



    for (int iNh = 0; iNh < Nh; ++iNh)
    {
        // register number : ePack
        TRANSPOSE_24X8_SAVE();
        srcPtr += loadTransposeStride;
    }

    //     MNN_PRINT("winograd in BT*D*B, transpose, loadTransposeStride:%zu, dstStep:%zu\n", loadTransposeStride, dstStep);
    // formatMatrix((const float*)srcBlock, {Nh, static_cast<int>(packCUnit), ePack});

    srcPtr = srcBlock;
    float* dstPtr = dstStart;
    for (int i4c = 0; i4c < packCUnit; ++i4c)
    {
        VecType s00 = VecType::load(srcPtr + 0 * loadTransposeStride + 0 * packCUnit);
        VecType s01 = VecType::load(srcPtr + 0 * loadTransposeStride + 1 * packCUnit);
        VecType s02 = VecType::load(srcPtr + 0 * loadTransposeStride + 2 * packCUnit);

        VecType s10 = VecType::load(srcPtr + 1 * loadTransposeStride + 0 * packCUnit);
        VecType s11 = VecType::load(srcPtr + 1 * loadTransposeStride + 1 * packCUnit);
        VecType s12 = VecType::load(srcPtr + 1 * loadTransposeStride + 2 * packCUnit);

        VecType s20 = VecType::load(srcPtr + 2 * loadTransposeStride + 0 * packCUnit);
        VecType s21 = VecType::load(srcPtr + 2 * loadTransposeStride + 1 * packCUnit);
        VecType s22 = VecType::load(srcPtr + 2 * loadTransposeStride + 2 * packCUnit);

        VecType s30 = VecType::load(srcPtr + 3 * loadTransposeStride + 0 * packCUnit);
        VecType s31 = VecType::load(srcPtr + 3 * loadTransposeStride + 1 * packCUnit);
        VecType s32 = VecType::load(srcPtr + 3 * loadTransposeStride + 2 * packCUnit);

        VecType s40 = VecType::load(srcPtr + 4 * loadTransposeStride + 0 * packCUnit);
        VecType s41 = VecType::load(srcPtr + 4 * loadTransposeStride + 1 * packCUnit);
        VecType s42 = VecType::load(srcPtr + 4 * loadTransposeStride + 2 * packCUnit);

        VecType s50 = VecType::load(srcPtr + 5 * loadTransposeStride + 0 * packCUnit);
        VecType s51 = VecType::load(srcPtr + 5 * loadTransposeStride + 1 * packCUnit);
        VecType s52 = VecType::load(srcPtr + 5 * loadTransposeStride + 2 * packCUnit);

        VecType s60 = VecType::load(srcPtr + 6 * loadTransposeStride + 0 * packCUnit);
        VecType s61 = VecType::load(srcPtr + 6 * loadTransposeStride + 1 * packCUnit);
        VecType s62 = VecType::load(srcPtr + 6 * loadTransposeStride + 2 * packCUnit);

        VecType s70 = VecType::load(srcPtr + 7 * loadTransposeStride + 0 * packCUnit);
        VecType s71 = VecType::load(srcPtr + 7 * loadTransposeStride + 1 * packCUnit);
        VecType s72 = VecType::load(srcPtr + 7 * loadTransposeStride + 2 * packCUnit);

        // to-try: reorder complicated commpute of 8x8
        auto ep0 = s00 * 36.f - s20 * 49.f + s40 * 14.f - s60;
        auto ep1 = s01 * 36.f - s21 * 49.f + s41 * 14.f - s61;
        auto ep2 = s02 * 36.f - s22 * 49.f + s42 * 14.f - s62;
        VecType::save(dstPtr + 0 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 0 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 0 * dstStep + 2 * packCUnit, ep2);

        ep0 = (s10 + s20) * 36.f - (s30 + s40) * 13.f + (s50 + s60);
        ep1 = (s11 + s21) * 36.f - (s31 + s41) * 13.f + (s51 + s61);
        ep2 = (s12 + s22) * 36.f - (s32 + s42) * 13.f + (s52 + s62);
        VecType::save(dstPtr + 1 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 1 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 1 * dstStep + 2 * packCUnit, ep2);

        ep0 = (s20 - s10) * 36.f + (s30 - s40) * 13.f + (s60 - s50);
        ep1 = (s21 - s11) * 36.f + (s31 - s41) * 13.f + (s61 - s51);
        ep2 = (s22 - s12) * 36.f + (s32 - s42) * 13.f + (s62 - s52);
        VecType::save(dstPtr + 2 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 2 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 2 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10 * 18.f + s20 * 9.f - s30 * 20.f - s40 * 10.f + s50 * 2.f + s60;
        ep1 = s11 * 18.f + s21 * 9.f - s31 * 20.f - s41 * 10.f + s51 * 2.f + s61;
        ep2 = s12 * 18.f + s22 * 9.f - s32 * 20.f - s42 * 10.f + s52 * 2.f + s62;
        VecType::save(dstPtr + 3 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 3 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 3 * dstStep + 2 * packCUnit, ep2);

        ep0 = s20 * 9.f - s10 * 18.f + s30 * 20.f - s40 * 10.f - s50 * 2.f + s60;
        ep1 = s21 * 9.f - s11 * 18.f + s31 * 20.f - s41 * 10.f - s51 * 2.f + s61;
        ep2 = s22 * 9.f - s12 * 18.f + s32 * 20.f - s42 * 10.f - s52 * 2.f + s62;
        VecType::save(dstPtr + 4 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 4 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 4 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10 * 12.f + s20 * 4.f - s30 * 15.f - s40 * 5.f + s50 * 3.f + s60;
        ep1 = s11 * 12.f + s21 * 4.f - s31 * 15.f - s41 * 5.f + s51 * 3.f + s61;
        ep2 = s12 * 12.f + s22 * 4.f - s32 * 15.f - s42 * 5.f + s52 * 3.f + s62;
        VecType::save(dstPtr + 5 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 5 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 5 * dstStep + 2 * packCUnit, ep2);

        ep0 = s20 * 4.f - s10 * 12.f + s30 * 15.f - s40 * 5.f - s50 * 3.f + s60;
        ep1 = s21 * 4.f - s11 * 12.f + s31 * 15.f - s41 * 5.f - s51 * 3.f + s61;
        ep2 = s22 * 4.f - s12 * 12.f + s32 * 15.f - s42 * 5.f - s52 * 3.f + s62;
        VecType::save(dstPtr + 6 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 6 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 6 * dstStep + 2 * packCUnit, ep2);

        ep0 = s30 * 49.f - s10 * 36.f - s50 * 14.f + s70;
        ep1 = s31 * 49.f - s11 * 36.f - s51 * 14.f + s71;
        ep2 = s32 * 49.f - s12 * 36.f - s52 * 14.f + s72;
        VecType::save(dstPtr + 7 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 7 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 7 * dstStep + 2 * packCUnit, ep2);
        srcPtr += ePack;
        dstPtr += ePack;
    }
}





static void destUnrollTransformUnit_sfc(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep,int T) {
    Vec8 s0 = Vec8::load(srcBlock + 0 * srcStep);
    Vec8 s1 = Vec8::load(srcBlock + 1 * srcStep);
    Vec8 s2 = Vec8::load(srcBlock + 2 * srcStep);
    Vec8 s3 = Vec8::load(srcBlock + 3 * srcStep);
    Vec8 s4 = Vec8::load(srcBlock + 4 * srcStep);
    Vec8 s5 = Vec8::load(srcBlock + 5 * srcStep);
    Vec8 s6 = Vec8::load(srcBlock + 6 * srcStep);
    Vec8 s7 = Vec8::load(srcBlock + 7 * srcStep);
    Vec8 s8 = Vec8::load(srcBlock + 8 * srcStep);
    Vec8 s9 = Vec8::load(srcBlock + 9 * srcStep);


    // std::cout<<"src 0"<<std::endl;
    // for(int i=0;i<10;i++)
    // {
    //     std::cout<<*(srcBlock+i*srcStep)<<" ";
    // }
    // std::cout<<std::endl;


    for (int i = 0; i < T - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        auto m0 = (s0 + s1*2.f - s2 - s3 + s4 + s5 - s6*2.f - s7)*0.166667 + s8 ;
        auto m1 = (s0 + s1     + s2 - s3*2.f - s4*2.f + s5 + s6 + s7)*0.166667  ;
        auto m2 = (s0 - s1     + s2*2.f - s3 + s4 - s5*2.f + s6 - s7)*0.166667  ;
        auto m3 = (s0 - s1*2.f + s2 + s3 + s4 + s5 - s6*2.f + s7)*0.166667  ;
        auto m4 = (s0 - s1     - s2 + s3*2.f - s4*2.f + s5 + s6 - s7)*0.166667  ;
        auto m5 = (s0 + s1     - s2*2.f + s3 + s4 - s5*2.f + s6 + s7)*0.166667 + s9  ;

        s0 = Vec8::load(srcFloatPtr + 0 * srcStep);
        s1 = Vec8::load(srcFloatPtr + 1 * srcStep);
        s2 = Vec8::load(srcFloatPtr + 2 * srcStep);
        Vec8::save(dstFloatPtr + 0 * dstStep, m0);
        s3 = Vec8::load(srcFloatPtr + 3 * srcStep);
        Vec8::save(dstFloatPtr + 1 * dstStep, m1);
        s4 = Vec8::load(srcFloatPtr + 4 * srcStep);
        Vec8::save(dstFloatPtr + 2 * dstStep, m2);
        s5 = Vec8::load(srcFloatPtr + 5 * srcStep);
        Vec8::save(dstFloatPtr + 3 * dstStep, m3);
        s6 = Vec8::load(srcFloatPtr + 6 * srcStep);
        Vec8::save(dstFloatPtr + 4 * dstStep, m4);
        s7 = Vec8::load(srcFloatPtr + 7 * srcStep);
        Vec8::save(dstFloatPtr + 5 * dstStep, m5);

        // std::cout<<"dst "<<i<<std::endl;
        // for(int i=0;i<6;i++)
        // {
        //     std::cout<<*(dstFloatPtr+i*dstStep)<<" ";
        // }
        // std::cout<<std::endl;

        // std::cout<<"src "<<i+1<<std::endl;
        // for(int i_t=0;i_t<8;i_t++)
        // {
        //     std::cout<<*(srcFloatPtr+i_t*srcStep)<<" ";
        // }
        // std::cout<<std::endl;


    }

    auto dstFloatPtr = (float*)(dstStart + (T - 1) * dstRowStep);

    auto m0 = (s0 + s1*2.f - s2 - s3 + s4 + s5 - s6*2.f - s7)*0.166667 + s8 ;
    auto m1 = (s0 + s1     + s2 - s3*2.f - s4*2.f + s5 + s6 + s7)*0.166667  ;
    auto m2 = (s0 - s1     + s2*2.f - s3 + s4 - s5*2.f + s6 - s7)*0.166667  ;
    auto m3 = (s0 - s1*2.f + s2 + s3 + s4 + s5 - s6*2.f + s7)*0.166667  ;
    auto m4 = (s0 - s1     - s2 + s3*2.f - s4*2.f + s5 + s6 - s7)*0.166667  ;
    auto m5 = (s0 + s1     - s2 + s3 + s4 - s5*2.f + s6 + s7)*0.166667 + s9 ;

    Vec8::save(dstFloatPtr + 0 * dstStep, m0);
    Vec8::save(dstFloatPtr + 1 * dstStep, m1);
    Vec8::save(dstFloatPtr + 2 * dstStep, m2);
    Vec8::save(dstFloatPtr + 3 * dstStep, m3);
    Vec8::save(dstFloatPtr + 4 * dstStep, m4);
    Vec8::save(dstFloatPtr + 5 * dstStep, m5);

    // std::cout<<"dst "<<T-1<<std::endl;
    // for(int i=0;i<6;i++)
    // {
    //     std::cout<<*(dstFloatPtr+i*dstStep)<<" ";
    // }
    // std::cout<<std::endl;

}



static void destUnrollTransformUnit_wino(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep,int T) {

    // std::cout<<"src 0"<<std::endl;
    // for(int i=0;i<8;i++)
    // {
    //     std::cout<<*(srcBlock+i*srcStep)<<" ";
    // }
    // std::cout<<std::endl;

    Vec8 s0 = Vec8::load(srcBlock + 0 * srcStep);
    Vec8 s1 = Vec8::load(srcBlock + 1 * srcStep);
    Vec8 s2 = Vec8::load(srcBlock + 2 * srcStep);
    Vec8 s3 = Vec8::load(srcBlock + 3 * srcStep);
    Vec8 s4 = Vec8::load(srcBlock + 4 * srcStep);
    Vec8 s5 = Vec8::load(srcBlock + 5 * srcStep);
    Vec8 s6 = Vec8::load(srcBlock + 6 * srcStep);
    Vec8 s7 = Vec8::load(srcBlock + 7 * srcStep);
    for (int i = 0; i < T - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        Vec8 mid0, mid1, mid2, mid3, mid4, mid5;
        mid0 = s1 + s2;
        mid1 = s1 - s2;
        mid2 = s3 + s4;
        mid3 = s3 - s4;
        mid4 = s5 + s6;
        mid5 = s5 - s6;
        auto m0 = s0 + mid0 + mid2 + mid4;
        auto m1 = mid1 + mid3 * 2.f + mid5 * 3.f;
        auto m2 = mid0 + mid2 * 4.f + mid4 * 9.f;
        auto m3 = mid1 + mid3 * 8.f + mid5 * 27.f;
        s0 = Vec8::load(srcFloatPtr + 0 * srcStep);
        auto m4 = mid0 + mid2 * 16.f + mid4 * 81.f;
        s1 = Vec8::load(srcFloatPtr + 1 * srcStep);
        auto m5 = mid1 + mid3 * 32.f + mid5 * 243.f + s7;
        s2 = Vec8::load(srcFloatPtr + 2 * srcStep);

        Vec8::save(dstFloatPtr + 0 * dstStep, m0);
        s3 = Vec8::load(srcFloatPtr + 3 * srcStep);
        Vec8::save(dstFloatPtr + 1 * dstStep, m1);
        s4 = Vec8::load(srcFloatPtr + 4 * srcStep);
        Vec8::save(dstFloatPtr + 2 * dstStep, m2);
        s5 = Vec8::load(srcFloatPtr + 5 * srcStep);
        Vec8::save(dstFloatPtr + 3 * dstStep, m3);
        s6 = Vec8::load(srcFloatPtr + 6 * srcStep);
        Vec8::save(dstFloatPtr + 4 * dstStep, m4);
        s7 = Vec8::load(srcFloatPtr + 7 * srcStep);
        Vec8::save(dstFloatPtr + 5 * dstStep, m5);
        
        // std::cout<<"dst "<<i<<std::endl;
        // for(int i=0;i<6;i++)
        // {
        //     std::cout<<*(dstFloatPtr+i*dstStep)<<" ";
        // }
        // std::cout<<std::endl;

        // std::cout<<"src "<<i+1<<std::endl;
        // for(int i_t=0;i_t<8;i_t++)
        // {
        //     std::cout<<*(srcFloatPtr+i_t*srcStep)<<" ";
        // }
        // std::cout<<std::endl;
    }

    auto dstFloatPtr = (float*)(dstStart + (T - 1) * dstRowStep);

    Vec8 mid0, mid1, mid2, mid3, mid4, mid5;
    mid0 = s1 + s2;
    mid1 = s1 - s2;
    auto m0 = s0 + mid0;
    mid2 = s3 + s4;
    mid3 = s3 - s4;
    m0 = m0 + mid2;
    mid4 = s5 + s6;
    mid5 = s5 - s6;
    m0 = m0 + mid4;

    auto m1 = mid1 + mid3 * 2.f + mid5 * 3.f;
    auto m2 = mid0 + mid2 * 4.f + mid4 * 9.f;
    auto m3 = mid1 + mid3 * 8.f + mid5 * 27.f;
    auto m4 = mid0 + mid2 * 16.f + mid4 * 81.f;
    auto m5 = mid1 + mid3 * 32.f + mid5 * 243.f + s7;

    Vec8::save(dstFloatPtr + 0 * dstStep, m0);
    Vec8::save(dstFloatPtr + 1 * dstStep, m1);
    Vec8::save(dstFloatPtr + 2 * dstStep, m2);
    Vec8::save(dstFloatPtr + 3 * dstStep, m3);
    Vec8::save(dstFloatPtr + 4 * dstStep, m4);
    Vec8::save(dstFloatPtr + 5 * dstStep, m5);
    
    // std::cout<<"dst "<<5<<std::endl;
    // for(int i=0;i<6;i++)
    // {
    //     std::cout<<*(dstFloatPtr+i*dstStep)<<" ";
    // }
    // std::cout<<std::endl;
}







//#define MNN_WINOGRAD_PRINT_REDUCE_RATE
//#define MNN_WINO_TRANFORM_TEST_CLOSE
namespace MNN {

ConvolutionPackWinograd::ConvolutionPackWinograd(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output,
                                         Backend *b, const float *originWeight, size_t originWeightSize,
                                         const float *bias, size_t biasSize, WinogradConfig config)
    : ConvolutionWinogradImpl(convOp, b) {
    int unit = config.unit;
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int pack = core->pack, bytes = core->bytes;
    int weightBytes = bytes;
    if (0!=core->matmulBytes) {
        weightBytes = core->matmulBytes;
    }
    mResource.reset(new Resource);
    mResource->backend = b;

    mDestUnrollTransform.reset(new CoreFunctions::WinoUnrollDestTransFunc[CONVOLUTION_WINOGRAD_MAX_UNIT + 1],
        std::default_delete<CoreFunctions::WinoUnrollDestTransFunc[]>());

    if (!mResource->copyBiasAlign(bias, biasSize)) {
        MNN_ERROR("Not Enough Memory\n");
        mValid = false;
        return;
    }
    MNN_ASSERT(mCommon->kernelX() == mCommon->kernelY());

    int threadNumber = ((CPUBackend *)backend())->threadNumber();

    auto kernelSize = mCommon->kernelY();
    WinogradGenerater generator(unit, kernelSize, 1, true);

    auto wino_a = generator.A();
    auto wino_b = generator.B();
    auto wino_g = generator.G();
    MNN_PRINT("A=\n");
    MNN::Math::Matrix::print(wino_a.get());
    MNN_PRINT("B=\n");
    MNN::Math::Matrix::print(wino_b.get());
    MNN_PRINT("G=\n");
    MNN::Math::Matrix::print(wino_g.get());

    int ePack, hPack, lPack;
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);

    int alpha        = unit + kernelSize - 1;
    int alpha2       = alpha * alpha;
    mSourceTransformPack = core->chooseWinoSourceTransformPack(alpha, alpha, ePack, lPack, pack);
    mSourceUnrollTransform =  core->chooseWinoSourceUnrollTransform(alpha, alpha);
    core->chooseWinoDestUnrollTransform(mDestUnrollTransform.get(), CONVOLUTION_WINOGRAD_MAX_UNIT + 1, alpha, unit);

    int srcCount                       = input->channel();
    int outputCount                    = output->channel();
    auto ic4 = UP_DIV(srcCount, pack);
    auto oc4 = UP_DIV(outputCount, pack);
    //mTempBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, ePack, ic4 + oc4, pack * alpha2, bytes}));
    mTempBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, ePack, ic4, pack * 164, bytes}));//可能是错的
    //input(1*24*4*(8*100)*4)+output(1*24*4*(8*64)*4)=total(1*24*4*(8*164)*4)


    // mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, 2, alpha2, pack, bytes}));
    // mGemmMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, ePack * UP_DIV(srcCount, lPack) * lPack, bytes}));

    mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, (1 + ic4 * ePack), 80, pack, bytes})); // 1 means original small buffer of alpha2 * pack.
    //用于储存单个输入块的两个通道变换后的数据
    mGemmMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, alpha, ePack * UP_DIV(srcCount, pack) * pack, bytes}));

    //mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, (1 + ic4 * ePack), alpha2, pack, bytes})); // 1 means original small buffer of alpha2 * pack.
    //mGemmMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, alpha, ePack * UP_DIV(srcCount, pack) * pack, bytes}));



    mA = generator.A();
    mB = generator.B();

    // Transform Kernel
    auto G = generator.G();
    // replace Tensor::createDevice by Tensor::create and allocTransformWeight's alloc=true to avoid malloc by onAcquireBuffer
    std::shared_ptr<Tensor> sourceWeight(Tensor::create<float>(
        std::vector<int>{outputCount, srcCount, kernelSize, kernelSize}, (void *)originWeight, Tensor::CAFFE));
    auto tempWeight = generator.allocTransformWeight(sourceWeight.get(), lPack, hPack, true);
    std::cout<<"tempWeight shape is"<<std::endl;
    tempWeight->printShape();
    auto shape = tempWeight->shape();
    shape.push_back(weightBytes);
    mResource->mWeight.reset(Tensor::createDevice<uint8_t>(shape));
    mValid = backend()->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    generator.transformWeight(tempWeight.get(), sourceWeight.get(), true);
    /*
    float* tempWeight_ptr = tempWeight->host<float>();
    std::cout<<"tempWeight:[";
    for (int i = 0; i < 100; ++i) 
    {
        if((i%10)==0){std::cout<<"/n";}
        std::cout<<tempWeight_ptr[i * 32 * 32]<<"  ";
    }
    std::cout<<"]"<<std::endl;
    */
    if (weightBytes != 4) {
        core->MNNFp32ToLowp(tempWeight->host<float>(), mResource->mWeight->host<int16_t>(), tempWeight->elementSize());
    } else {
        ::memcpy(mResource->mWeight->host<float>(), tempWeight->host<float>(), tempWeight->size());
    }
    /*
    float* mWeight_ptr = mResource->mWeight->host<float>();
    std::cout<<"mWeight:[";
    for (int i = 0; i < 100; ++i) 
    {
        std::cout<<mWeight_ptr[i * 32 * 32]<<",";
    }
    std::cout<<"]";
    */
    mPostParameters = getPostParameters();
}
ConvolutionPackWinograd::~ConvolutionPackWinograd() {
    // Do nothing
}
bool ConvolutionPackWinograd::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvolutionPackWinograd(mResource, op->main_as_Convolution2D()->common(), bn);
    dstExe->mA = mA;
    dstExe->mB = mB;
    dstExe->mTempBuffer.reset(Tensor::createDevice<uint8_t>(mTempBuffer->shape()));
    dstExe->mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>(mTransformMidBuffer->shape()));
    dstExe->mGemmMidBuffer.reset(Tensor::createDevice<uint8_t>(mGemmMidBuffer->shape()));
    dstExe->mSourceTransformPack = mSourceTransformPack;
    dstExe->mSourceUnrollTransform = mSourceUnrollTransform;
    dstExe->mDestUnrollTransform = mDestUnrollTransform;
    dstExe->mPostParameters = mPostParameters;
    *dst = dstExe;
    return true;
}

ErrorCode ConvolutionPackWinograd::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_CONCURRENCY_BEGIN(tId, mMainFunction.first) {
        mMainFunction.second(tId, inputs[0]->host<uint8_t>(), outputs[0]->host<uint8_t>());
    };
    MNN_CONCURRENCY_END();

    MNN_CONCURRENCY_BEGIN(tId, mPostFunction.first) {
        mPostFunction.second(tId, outputs[0]->host<uint8_t>());
    };
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

WinogradConfig ConvolutionPackWinograd::bestWinogradUnit(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b, const PerfConfig& denseConfig) {

    // compare cost value
    WinogradConfig wconfig;


    auto core = static_cast<CPUBackend*>(b)->functions();
    auto winogradMemoryLevel = static_cast<CPUBackend*>(b)->getRuntime()->hint().winogradMemoryUsed;
    int multiBytes = static_cast<CPUBackend*>(b)->functions()->bytes;
    if (static_cast<CPUBackend*>(b)->functions()->matmulBytes != 0) {
        multiBytes = static_cast<CPUBackend*>(b)->functions()->matmulBytes;
    }
    int ow      = outputTensor->width();
    int oh      = outputTensor->height();
    int oc      = outputTensor->channel();
    int ePack, hPack, lPack;
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    int unit2   = UP_DIV(ow * oh, threadNumber);

    int maxUnit = (int)::sqrtf((float)unit2);
    maxUnit     = std::min(maxUnit, CONVOLUTION_WINOGRAD_MAX_UNIT);
    maxUnit     = std::max(maxUnit, CONVOLUTION_WINOGRAD_MIN_UNIT);
    if (winogradMemoryLevel != 3) {
       maxUnit = CONVOLUTION_WINOGRAD_MIN_UNIT;
    }

    int ic           = inputTensor->channel();
    auto kernelSize  = common->kernelY();
    int unit         = 0;
    float maxRate    = 0.0f;
    float originCost = (float)ow * oh * (2.0 * ic) * oc * kernelSize * kernelSize; // macs, with bias
    std::set<int> supportSu{4, 6, 8};
    if (multiBytes < 4) {
        supportSu = {4, 6};
    }
    CoreFunctions::WinoUnrollDestTransFunc destTransform[CONVOLUTION_WINOGRAD_MAX_UNIT + 1];
    for (int u = CONVOLUTION_WINOGRAD_MIN_UNIT; u <= maxUnit; ++u) {
        auto sui = u + kernelSize - 1;
        auto su = (float)sui;
        if (supportSu.find(sui) == supportSu.end()) {
            continue;
        }
        core->chooseWinoDestUnrollTransform(destTransform, CONVOLUTION_WINOGRAD_MAX_UNIT + 1, sui, u);
            if (nullptr == destTransform[sui]) {
            continue;
        }
        // /*Let F(6,3) be choosed when it can speed up from F(2,3) than 0.6*/

        // float penalty = (su * su) / (float)(kernelSize * kernelSize) * 0.12f;
        // float winogradCost =
        //     (2 * su * su * ic + su * su * ic * oc + (su + u) * u * oc) * 2 * (UP_DIV(ow, u) * UP_DIV(oh, u));
        // float reduceRate = originCost / winogradCost - penalty;

        // new metrics for winograd, only need to calculate absolute compute complexity.
        // add instructions are about (n - 2), multiply operations are (n - 4). as a result operations are (2n - 6).
        float winogradCost =
            ( (2 * su) * su * su * ic + 2 * su * su * ic * oc + ((su + u) * u * (2 * su) * oc)) * (UP_DIV(ow, u) * UP_DIV(oh, u));
        float reduceRate = originCost / winogradCost;

        // MNN_PRINT("ow=%d, oh=%d, winogradCost:%f, reduceRate:%f, winograd unit:%d\n", ow, oh, winogradCost, reduceRate, u);
        if (reduceRate > maxRate) {
            maxRate = reduceRate;
            unit    = u;
        }
    }
    if (maxRate < 1.0f) {
        wconfig.unit = 0;
        return wconfig;
    }
    wconfig.unit = unit;
    return wconfig;
}

ErrorCode ConvolutionPackWinograd::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    int threadNumber = ((CPUBackend*)(backend()))->threadNumber();
    mTempBuffer->setLength(0, threadNumber);
    mGemmMidBuffer->setLength(0, threadNumber);
    mTransformMidBuffer->setLength(0, threadNumber);
    // FUNC_PRINT(mA->length(1));
    bool success = backend()->onAcquireBuffer(mTempBuffer.get(), Backend::DYNAMIC);
    success      = success && backend()->onAcquireBuffer(mGemmMidBuffer.get(), Backend::DYNAMIC);
    success      = success && (backend()->onAcquireBuffer(mTransformMidBuffer.get(), Backend::DYNAMIC));
    backend()->onReleaseBuffer(mTempBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTransformMidBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mGemmMidBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int pack = core->pack, bytes = core->bytes;

    auto input   = inputs[0];
    auto output  = outputs[0];
    auto dstUnit = mA->length(1); // m
    auto srcUnit = mA->length(0); // n
    auto midUnit = 10;
    int ePack, lPack, hPack;
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);

    auto srcUnit2 = srcUnit * srcUnit;
    auto alphaXStride = srcUnit * ePack * pack;
    auto IC4alpha2Stride = 80 * ePack * pack;

    int ow   = output->width();
    int oh   = output->height();
    int iw   = input->width();
    int ih   = input->height();
    int ic_4 = UP_DIV(input->channel(), pack);
    int dc_4 = UP_DIV(output->channel(), pack);
    int batch = input->batch();
    // MNN_PRINT("%d, %d\n", srcUnit, dstUnit);

    int padY = mPadY;
    int padX = mPadX;

    auto wUnit = UP_DIV(ow, dstUnit); // ow / m
    auto hUnit = UP_DIV(oh, dstUnit); // oh / m

    auto totalCount   = wUnit * hUnit * batch;
    // MNN_PRINT("ow=%d, oh=%d\n", ow, oh);
    
    std::vector<int> divides(threadNumber+1);
    static_cast<const CPURuntime*>( static_cast<CPUBackend*>(backend())->getRuntime())->computeDivideSizes(totalCount, divides.data()+1);
    divides[0] = 0;
    auto midBuffer0Bytes = (8*10) * pack * bytes;
    bool allow_x86_bf16_winograd = true;
#ifdef MNN_USE_SSE
    allow_x86_bf16_winograd = bytes != 2; // only bf16 has length of 2 byte on x86. fp16 dosnot exist.
#endif

    auto weight    = mResource->mWeight->host<uint8_t>();
    auto bias      = mResource->mBias->host<uint8_t>();
    mMainFunction.first = threadNumber;
    mMainFunction.second = [=](int tId, const uint8_t* inputOrigin, uint8_t* dstOrigin) 
    {
        int srcunit = 8; int midunit = 10; int dstunit = 6;
        int ic_4=4;int epack=24;int pack = 8;
        //权重变换
        int outputCount = 32; int srcCount = 32; int kernelSize = 3;
        float originWeight[outputCount*srcCount*kernelSize*kernelSize];
        for(int i =0;i<outputCount*srcCount*kernelSize*kernelSize;i++)
        {
            originWeight[i] = 1;
        }
        std::shared_ptr<Tensor> sourceWeight(Tensor::create<float>(
        std::vector<int>{outputCount, srcCount, kernelSize, kernelSize}, (void *)originWeight, Tensor::CAFFE));
        auto tempWeight_wino = std::shared_ptr<Tensor>(Tensor::create<float>({srcunit*srcunit, 8, 32, 1, 4}));
        auto tempWeight_sfc = std::shared_ptr<Tensor>(Tensor::create<float>({midunit*midunit, 8, 32, 1, 4}));
        
        int unit = 6;
        WinogradGenerater generator(unit, kernelSize, 1, true);
        mA = generator.A();
        mB = generator.B();
        // Transform Kernel
        auto G = generator.G();

        WeightTransSFC(tempWeight_sfc.get(), sourceWeight.get(), true);

        std::ofstream outfile_tempWeight_sfc("../tempWeight_sfc.txt"); // 指定文件名

        for(int i =0;i<10*10;i++)
        {
            outfile_tempWeight_sfc<<(tempWeight_sfc->host<float>())[i*32*32]<<",";
            if(((i+1)%10)==0){outfile_tempWeight_sfc<<std::endl;}
        }

        generator.transformWeight(tempWeight_wino.get(), sourceWeight.get(), true);

        std::ofstream outfile_tempWeight_wino("../tempWeight_wino.txt"); // 指定文件名

        for(int i =0;i<8*8;i++)
        {
            outfile_tempWeight_wino<<(tempWeight_wino->host<float>())[i*32*32]<<",";
            if(((i+1)%8)==0){outfile_tempWeight_wino<<std::endl;}
        }
    
        //源数据第一次变换
        float src_data[ic_4*epack*srcunit*srcunit*pack];//ic4*epack*srcunit^2*pack
        for(int i=0;i<4*24*8*8*8;i++)
        {
            src_data[i] = 1 ;
        }
        
        float midBuffer_wino[ic_4*srcunit*srcunit*epack*pack];
        float midBuffer_sfc[ic_4*epack*midunit*srcunit*pack];
        int sourceZtep = epack*srcunit*srcunit*pack;
        int destSOffset = 0;

        auto time_1 = std::chrono::high_resolution_clock::now();
        for(int i=0;i<24;i++)
        {
            float* src_ptr = src_data + i*8*8*8;
            auto midBuffer_wino_Offset = midBuffer_wino + destSOffset;
            for(int z = 0;z<4;z++)
            {
                float* srcZ_ptr = src_ptr + z*sourceZtep;
                auto midBuffer_wino_Z = midBuffer_wino_Offset + z*epack*srcunit*srcunit*pack;
                //const float* srcBlock, float* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep
                sourceUnrollTransform_wino((const float*)src_ptr, (float*)midBuffer_wino_Z, srcUnit * pack, ePack * pack, pack, alphaXStride);
            }
            destSOffset += pack;
        }
        auto time_2 = std::chrono::high_resolution_clock::now();
        destSOffset = 0;
        for(int i=0;i<24;i++)
        {
            float* src_ptr = src_data + i*8*8*8;
            auto midBuffer_sfc_Offset = midBuffer_sfc + destSOffset;
            for(int z = 0;z<4;z++)
            {
                float* srcZ_ptr = src_ptr + z*sourceZtep;
                auto midBuffer_sfc_Z  = midBuffer_sfc_Offset  + z*epack*midunit*srcunit*pack;
                sourceUnrollTransform_sfc((const float*)src_ptr, (float*)midBuffer_sfc_Z, srcunit * pack, ePack * pack, pack, alphaXStride);
                //const float* srcBlock, float* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep
            }
            destSOffset += pack;
        }
        auto time_3 = std::chrono::high_resolution_clock::now();

        auto duration_wino = std::chrono::duration_cast<std::chrono::microseconds>(time_2 - time_1);
        auto duration_sfc = std::chrono::duration_cast<std::chrono::microseconds>(time_3 - time_2);
        std::cout << "SFC execution time: " << duration_sfc.count() << " microseconds" << std::endl;
        std::cout << "WINO execution time: " << duration_wino.count() << " microseconds" << std::endl;
        {
            // auto midTransformPtr_float = (float*)midBuffer_sfc;
            // std::ofstream midBuffer_sfc_outfile("../midBuffer_sfc.txt"); // 指定文件名
            // for(int c=0;c<4;c++)
            // {
            //     for(int i=0;i<10*8*24;i++)
            //     {
            //         for(int j=0;j<8;j++)
            //         {
            //             midBuffer_sfc_outfile<<midTransformPtr_float[i*8+j]<<",";
            //         }
            //         midBuffer_sfc_outfile<<std::endl;
            //     }
            // }

            // midTransformPtr_float = (float*)midBuffer_wino;
            // std::ofstream midBuffer_wino_outfile("../midBuffer_wino.txt"); // 指定文件名
            // for(int c=0;c<4;c++)
            // {
            //     for(int i=0;i<8*8*24;i++)
            //     {
            //         for(int j=0;j<8;j++)
            //         {
            //             midBuffer_wino_outfile<<midTransformPtr_float[i*8+j]<<",";
            //         }
            //         midBuffer_wino_outfile<<std::endl;
            //     }
            // }


            // return;
        }

        //源数据第二次变换和与权重逐元素相乘
        float mGemmMidBuffer_sfc  [midunit*ic_4*pack*epack];
        float mGemmMidBuffer_wino [srcunit*ic_4*pack*epack];
        
        int oc_4 = 4;
        float mTempBuffer_sfc    [midunit*midunit*oc_4*epack*pack];
        float mTempBuffer_wino   [srcunit*srcunit*oc_4*epack*pack];

        std::vector<size_t> parameters(6)   ;
        parameters[1] = input->channel()    ;
        parameters[2] = output->channel()   ;
        parameters[4] = 0                   ;
        parameters[5] = 0                   ;
        parameters[0] = 0                   ;
        parameters[3] = ePack * pack * bytes;

        auto time_4 = std::chrono::high_resolution_clock::now();

        for (int iNw = 0; iNw < srcUnit; ++iNw) //wino
        { // i_Nw
            auto midTransformPtr = midBuffer_wino + iNw * srcUnit * ePack * pack;
            auto unitsGemmbuffer = mGemmMidBuffer_wino;
            for (int z = 0; z < ic_4; ++z) 
            { // ic_4
                sourceTransformUnitPack24_wino((float*)midTransformPtr, (float*)unitsGemmbuffer, ePack * pack * ic_4);
                unitsGemmbuffer += ePack * pack;
                midTransformPtr += srcunit*srcunit*epack*pack;
            }
            // Previous tranform requires xC aligned with EPack, xC should be Epack;
            for (int iNh = 0; iNh < srcUnit; ++iNh) 
            { // i_Nh, gemm
                auto unitsGemmbuffer = mGemmMidBuffer_wino + iNh * ic_4 * pack * ePack;
                auto _dstFloatPtr = (float*)(mTempBuffer_wino + (iNh * srcunit + iNw) * dc_4 * pack * ePack);
                auto _weightFloatPtr = (const float*)(tempWeight_wino->host<uint8_t>() + (iNh * srcunit + iNw) * 4096);
                core->MNNPackedMatMul(_dstFloatPtr, (float*)unitsGemmbuffer, _weightFloatPtr, parameters.data(), nullptr, nullptr, nullptr, nullptr);
            }
        }

        auto time_5 = std::chrono::high_resolution_clock::now();

        for (int iNw = 0; iNw < midunit; ++iNw) //sfc
        { // i_Nw
            auto midTransformPtr = midBuffer_sfc + iNw * alphaXStride;
            auto unitsGemmbuffer = mGemmMidBuffer_sfc;
            for (int z = 0; z < ic_4; ++z) 
            { // ic_4
                sourceTransformUnitPack24_sfc((float*)midTransformPtr, (float*)unitsGemmbuffer, ePack * pack * ic_4);
                unitsGemmbuffer += ePack * pack;
                midTransformPtr += midunit*srcunit*epack*pack;
            }

            // Previous tranform requires xC aligned with EPack, xC should be Epack;
            for (int iNh = 0; iNh < midunit; ++iNh) 
            { // i_Nh, gemm
                auto unitsGemmbuffer = mGemmMidBuffer_sfc + iNh * ic_4 * pack * ePack;
                auto _dstFloatPtr = (float*)(mTempBuffer_sfc + (iNh * midunit + iNw) * dc_4 * pack * ePack);
                auto _weightFloatPtr = (const float*)(tempWeight_sfc->host<uint8_t>() + (iNh * midunit + iNw) * 4096);

                core->MNNPackedMatMul(_dstFloatPtr, (float*)unitsGemmbuffer, _weightFloatPtr, parameters.data(), nullptr, nullptr, nullptr, nullptr);
            }
        }

        auto time_6 = std::chrono::high_resolution_clock::now();

        auto duration_wino_2 = std::chrono::duration_cast<std::chrono::microseconds>(time_5 - time_4);
        auto duration_sfc_2 = std::chrono::duration_cast<std::chrono::microseconds>(time_6 - time_5);
        std::cout << "SFC_2 execution time: " << duration_sfc_2.count() << " microseconds" << std::endl;
        std::cout << "WINO_2 execution time: " << duration_wino_2.count() << " microseconds" << std::endl;

        std::ofstream outfile_mTempBuffer_wino("../mTempBuffer_wino.txt"); // 指定文件名

        for(int i =0;i<8*8;i++)
        {
            outfile_mTempBuffer_wino<<mTempBuffer_wino[i*oc_4*epack*pack]<<" ";
            if(((i+1)%8)==0){outfile_mTempBuffer_wino<<std::endl;}
        }



        //目标变换
        auto dstZStep = epack*dstunit*dstunit*pack;
        auto srcZStep = epack * pack;
        auto unitStep = epack * oc_4 * pack;

        float dst_mid_wino [dstunit*srcunit*pack];
        float dst_wino     [oc_4*epack*dstunit*dstunit*pack];

        float dst_mid_sfc [dstunit*midunit*pack];
        float dst_sfc     [oc_4*epack*dstunit*dstunit*pack];
        
        auto time_7 = std::chrono::high_resolution_clock::now();

        for(int si =0;si<24;si++)
        {
            auto srcXi = mTempBuffer_wino + pack * si;
            auto dstStart = dst_wino + si*dstunit*dstunit*pack;
            for(int z=0;z<oc_4;z++)
            {

                auto srcZ     = srcXi + z * srcZStep;
                auto dstZAddr = dstStart + z * dstZStep;

                destUnrollTransformUnit_wino((const float*)srcZ, (float*)dst_mid_wino, nullptr, nullptr, unitStep, dstunit * pack, srcunit * unitStep, pack,8);
                destUnrollTransformUnit_wino((const float*)dst_mid_wino, (float*)dstZAddr,  nullptr, nullptr, pack, pack * dstunit, pack * dstunit, pack,6);
                //const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep,int T

            }
        }

        auto time_8 = std::chrono::high_resolution_clock::now();


        for(int si =0;si<24;si++)
        {
            auto srcXi = mTempBuffer_sfc + pack * si;
            auto dstStart = dst_sfc + si*dstunit*dstunit*pack;
            for(int z=0;z<oc_4;z++)
            {
                auto srcZ     = srcXi + z * srcZStep;
                auto dstZAddr = dstStart + z * dstZStep;

                destUnrollTransformUnit_sfc((const float*)srcZ, (float*)dst_mid_sfc, nullptr, nullptr, unitStep, dstunit * pack, srcunit * unitStep, pack,10);
                destUnrollTransformUnit_sfc((const float*)dst_mid_sfc, (float*)dstZAddr,  nullptr, nullptr, pack, pack * dstunit, pack * dstUnit, pack,6);

            }
        }

        auto time_9 = std::chrono::high_resolution_clock::now();

        auto duration_wino_3 = std::chrono::duration_cast<std::chrono::microseconds>(time_8 - time_7);
        auto duration_sfc_3 = std::chrono::duration_cast<std::chrono::microseconds>(time_9 - time_8);
        std::cout << "SFC_3 execution time: " << duration_sfc_3.count() << " microseconds" << std::endl;
        std::cout << "WINO_3 execution time: " << duration_wino_3.count() << " microseconds" << std::endl;

        // for(int i =0;i<oc_4*epack*dstunit*dstunit*pack;i++)
        // {
        //     std::cout<<dst_sfc[i]<<" ";
        // }



    };
    std::vector<int> postDivides(threadNumber+1);
    static_cast<const CPURuntime*>( static_cast<CPUBackend*>(backend())->getRuntime())->computeDivideSizes(dc_4, postDivides.data()+1);
    postDivides[0] = 0;

    mPostFunction.first = threadNumber;
    mPostFunction.second = [=](int tId, uint8_t* outputOrigin) {
        auto dstOrigin = outputOrigin;
        int tSta = postDivides[tId];
        int tFin = postDivides[tId+1];
        for (int dy=tSta; dy < tFin; ++dy) {
            auto dataFloatPtr = (float*)(dstOrigin + ow * oh * batch * dy * pack * bytes);
            auto biasFloatPtr = (const float*)(bias + pack * dy * bytes);
            core->MNNAxByClampBroadcastUnit(dataFloatPtr, dataFloatPtr, biasFloatPtr, ow * oh * batch, 0, 0, 1,  mPostParameters.data());
        }
    };
    return NO_ERROR;
}

} // namespace MNN
