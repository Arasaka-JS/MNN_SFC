#include <math.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <MNN/Interpreter.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <stdlib.h>
#include <string.h>
#include <MNN/MNNDefine.h>
#include "math/Matrix.hpp"
#include "math/WingoradGenerater.hpp"
#include <math.h>
#include <random>
#include <MNN/expr/ExprCreator.hpp>



using namespace MNN;

void printTensorInfo(MNN::Tensor* tensor) {
    // 打印 Tensor 的基本信息
    std::cout << "Tensor dimensions: " << tensor->dimensions() << std::endl;
    std::cout << "Tensor shape: ";
    for (int i = 0; i < tensor->dimensions(); ++i) {
        std::cout << tensor->shape()[i] << " ";
    }
    std::cout << std::endl;

    // 打印 Tensor 数据的前几个值
    std::cout << "Tensor data (first 10 elements): ";
    float* data = tensor->host<float>();
    for (int i = 0; i < 10 && i < tensor->elementSize(); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}


int main()
{
    auto model ="/home/skw404/MNN_SFC/MNN/SFC/Test00/build/CNN_0909.mnn";
    
    // create net and session
    auto mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model));
    MNN::ScheduleConfig netConfig;
    netConfig.type      = MNN_FORWARD_CPU;
    netConfig.numThread = 1;
    auto session        = mnnNet->createSession(netConfig);

    auto input = mnnNet->getSessionInput(session, nullptr);
    auto output = mnnNet->getSessionOutput(session, nullptr);
    
    MNN::Tensor* inHostTensor = MNN::Tensor::create <float>(input->shape(), NULL, MNN::Tensor::CAFFE);


    for(int i = 0; i < inHostTensor->elementSize(); i++) 
    {

        inHostTensor->host<float>()[i] = 1;
    }

    input->copyFromHostTensor(inHostTensor);

    //运行网络

    mnnNet->runSession(session);

    MNN::Tensor* outHostTensor = MNN::Tensor::create <float>(output->shape(), NULL, MNN::Tensor::CAFFE);

    output->copyToHostTensor(outHostTensor);

    printTensorInfo(outHostTensor);



}