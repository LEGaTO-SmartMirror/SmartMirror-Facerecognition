/* 
 *  File: FaceDetector.h
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

#include "SORT.h"
#include "TensorflowHelper.h"
#include "Types.h"
#include "Utils.h"

#include <vector>

#include <opencv2/opencv.hpp>

DEFINE_EXCEPTION(FaceDetectorException);

class FaceDetector : public TensorflowHelper
{
	inline static const double MEMORY_FRACTION  = 0.15;
	inline static const std::string GRAPH_MODEL = "modules/SmartMirror-Facerecognition/FaceRecognition_CMake/model/frozen_inference_graph_face.pb";
	inline static const std::string INPUT_LAYER = "image_tensor:0";
	inline static const std::vector<std::string> OUTPUT_LAYERS{
		"detection_boxes:0",
		"detection_scores:0",
		"detection_classes:0",
		"num_detections:0"
	};

public:
	FaceDetector() :
		m_trackerSort(25, 3),
		m_verbose(false)
	{
		// First we load and initialize the model.
		tensorflow::Status status = LoadGraph(GRAPH_MODEL, MEMORY_FRACTION);

		if (!status.ok()) throw(FaceDetectorException(status.ToString()));
	}

	TrackingObjects Detect(const std::string& fileName, cv::Mat* pImageMat = nullptr, const float& scoreThreshold = 0.15f)
	{
		cv::Mat image = cv::imread(fileName);
		if (pImageMat) *pImageMat = image.clone();
		return Detect(image, scoreThreshold);
	}

	TrackingObjects Detect(const cv::Mat& image, const float& scoreThreshold = 0.15f)
	{
		const tensorflow::Tensor& inputTensor = LoadTensorFromImage<>(image);

		// Actually run the image through the model.
		std::vector<tensorflow::Tensor> outputs;
		tensorflow::Status status = m_session->Run({ { INPUT_LAYER, inputTensor } }, OUTPUT_LAYERS, {}, &outputs);

		if (!status.ok()) throw(FaceDetectorException("Running model failed: " + status.ToString()));

		float* pBoxes = outputs[0].flat<float>().data();
		float* pScore = outputs[1].flat<float>().data();

		TrackingObjects trackingDets;
		int32_t imgW = image.size().width;
		int32_t imgH = image.size().height;

		for (int64_t i = 0; i < outputs[0].NumElements(); i += 4)
		{
			// std::cout << pScore[i / 4] << std::endl;
			if (pScore[i / 4] > scoreThreshold)
			{
				int32_t s = pScore[i / 4] * 100;

				if (typeid(BBox) == typeid(cv::Rect))
				{
					int32_t x = pBoxes[i + 1] * imgW;
					int32_t y = pBoxes[i + 0] * imgH;
					int32_t w = pBoxes[i + 3] * imgW;
					int32_t h = pBoxes[i + 2] * imgH;

					if (x < 0) x = 0;
					if (y < 0) y = 0;
					if (w > imgW - 1) w = imgW - 1;
					if (h > imgH - 1) h = imgH - 1;
					trackingDets.push_back(TrackingObject(BBox(x, y, w, h), s));
				}
				else
				{
					float x = pBoxes[i + 1];
					float y = pBoxes[i + 0];
					float w = pBoxes[i + 3];
					float h = pBoxes[i + 2];
					trackingDets.push_back(TrackingObject(BBox(x, y, w, h), s));
				}
			}
		}

		return trackingDets;
	}

	TrackingObjects FindFacesTracked(const std::string& fileName, cv::Mat* pImageMat = nullptr, const float& scoreThreshold = 0.15f)
	{
		cv::Mat image = cv::imread(fileName);
		if (pImageMat) *pImageMat = image.clone();
		return FindFacesTracked(image, scoreThreshold);
	}

	TrackingObjects FindFacesTracked(const cv::Mat& image, const float& scoreThreshold = 0.15f)
	{
		TrackingObjects trackingDets = Detect(image, scoreThreshold);

		int32_t width  = image.size().width;
		int32_t height = image.size().height;

		if (m_verbose)
		{
			for (const TrackingObject& box : trackingDets)
				std::cout << "FindFaces: x=" << box.bBox.x << "; y=" << box.bBox.y << "; w=" << box.bBox.width << "; h=" << box.bBox.height << "; s=" << box.score << std::endl;
		}

		TrackingObjects trackers = m_trackerSort.Update(trackingDets);

		for (TrackingObject& box : trackers)
		{
			if (box.bBox.x < 0) box.bBox.x = 0;
			if (box.bBox.y < 0) box.bBox.y = 0;
			if (box.bBox.width > width - 1) box.bBox.width = width - 1;
			if (box.bBox.height > height - 1) box.bBox.height = height - 1;
		}

		return trackers;
	}

	void ResetTracker()
	{
		m_trackerSort.ResetCounter();
	}

	void SetVerbose(const bool& en = true)
	{
		m_verbose = en;
	}

private:
	SORT m_trackerSort;
	bool m_verbose;
};
