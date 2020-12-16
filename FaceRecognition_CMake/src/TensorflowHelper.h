/* 
 *  File: TensorflowHelper.h
 *  Copyright (c) 2020 Florian Porrmann
 *  
 *  MIT License
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *  
 */

#pragma once

#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session.h>

#include <opencv2/core/types.hpp>

class TensorflowHelper
{
protected:
	TensorflowHelper() :
		m_session()
	{
	}

	tensorflow::Status LoadGraph(const std::string& graphFileName, const double& memoryFraction = 1.0)
	{
		// std::cout << "Loading Graph: " << graphFileName << " with memoryFraction=" << memoryFraction << std::endl;
		tensorflow::GraphDef graphDef;
		tensorflow::Status status = ReadBinaryProto(tensorflow::Env::Default(), graphFileName, &graphDef);

		if (!status.ok())
			return tensorflow::errors::NotFound("Failed to load compute graph at '", graphFileName, "'");

		tensorflow::SessionOptions options;
		options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(memoryFraction);
		options.config.mutable_gpu_options()->set_allow_growth(false);

		m_session.reset(tensorflow::NewSession(options));

		status = m_session->Create(graphDef);
		if (!status.ok()) return status;

		return tensorflow::Status::OK();
	}

	template<typename T = uint8_t>
	tensorflow::Tensor LoadTensorFromImage(const cv::Mat& image, const tensorflow::DataType& tensorType = tensorflow::DT_UINT8, const int32_t& ocvType = CV_8UC3)
	{
		cv::Size imageSize = image.size();

		tensorflow::Tensor inputTensor(tensorType, tensorflow::TensorShape({ 1, imageSize.height, imageSize.width, 3 }));
		T* pTensorData = inputTensor.flat<T>().data();
		cv::Mat cameraImage(imageSize.height, imageSize.width, ocvType, pTensorData);
		image.convertTo(cameraImage, ocvType);
		return inputTensor;
	}

protected:
	std::unique_ptr<tensorflow::Session> m_session;
};