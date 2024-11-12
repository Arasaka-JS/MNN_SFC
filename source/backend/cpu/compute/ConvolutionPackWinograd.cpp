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
#include <MNN/AutoTime.hpp>
#include "core/MemoryFormater.h"
#include <iostream>

constexpr int FULSE_THRESHHOLD_NUMERATOR = 10;
constexpr int FULSE_THRESHHOLD_DENOMINATOR = 10;

using namespace MNN::Math;

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
        //数据准备


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
