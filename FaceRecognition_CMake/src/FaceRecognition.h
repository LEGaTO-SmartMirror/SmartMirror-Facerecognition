/* 
 *  File: FaceRecognition.h
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

#include "FaceDetector.h"
#include "TensorflowHelper.h"
#include "Timer.h"

#include <opencv2/img_hash/phash.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

#include <experimental/filesystem>
#include <fstream>

namespace fs = std::experimental::filesystem;

DEFINE_EXCEPTION(FaceRecognitionException);

class FaceRecognition : public TensorflowHelper
{
	struct Face
	{
		cv::Mat rep;
		int32_t identity;
	};

	inline static const double MEMORY_FRACTION        = 0.15;
	inline static const std::string GRAPH_MODEL       = "../model/facenet.pb";
	inline static const std::string SVM_MODEL         = "../model/svm.yaml";
	inline static const std::string PEOPLE_FILE       = "../model/people.txt";
	inline static const std::string INPUT_LAYER       = "input:0";
	inline static const std::string PHASE_TRAIN_LAYER = "phase_train:0";
	inline static const std::vector<std::string> OUTPUT_LAYERS{ "embeddings:0" };

	inline static const std::string TRAINING_DATA_FOLDER = "../facedatabase/images";
	inline static const uint32_t FACE_WIDTH              = 160;
	inline static const uint32_t FACE_HEIGHT             = 160;
	inline static const uint32_t FACE_CHANNEL            = 3;

public:
	FaceRecognition(const bool& trainMode = false) :
		m_svm(),
		m_verbose(false)
	{
		// First we load and initialize the model.
		tensorflow::Status status = LoadGraph(GRAPH_MODEL, MEMORY_FRACTION);

		if (!status.ok()) throw(FaceDetectorException(status.ToString()));

		// In training mode only the tensorflow graph needs to be loaded,
		// eveything else will be created during the training
		if (trainMode) return;

		m_svm = cv::ml::SVM::load(SVM_MODEL);
		loadPeople();
	}

	std::string IdentifyFaceInBB(const cv::Mat& image, const TrackingObject& box)
	{
		return IdentifyFaceInBB(image, box.bBox);
	}

	std::string IdentifyFaceInBB(const cv::Mat& image, const BBox& bBox)
	{
		return IdentifyFace(CropAndAlignFace(image, bBox));
	}

	std::string IdentifyFace(const cv::Mat& face)
	{
		if (face.size() != cv::Size(FACE_WIDTH, FACE_HEIGHT)) return string_format("Unknown - Wrong Size (%dx%d)", face.size().width, face.size().height);

		cv::Mat rep = recognize(face);

		uint32_t id = static_cast<uint32_t>(m_svm->predict(rep));

		if (m_people.count(id) < 1) return "Unknown - Not in List";

		return m_people[id];
	}

	static void LearnFromPics()
	{
		FaceDetector detector;
		FaceRecognition fr(true);
		std::map<std::string, Face> faces;
		std::ofstream peopleOut(PEOPLE_FILE);
		Timer fullTimer;
		Timer dirTimer;

		if (!peopleOut.is_open())
			throw(FaceRecognitionException(string_format("Unable to create people file: %s", PEOPLE_FILE)));

		fullTimer.Start();

		for (const fs::directory_entry& entry : fs::directory_iterator(TRAINING_DATA_FOLDER))
		{
			if (fs::is_directory(entry))
			{
				dirTimer.Start();
				std::string dirName           = entry.path().filename().string();
				std::vector<std::string> args = splitString(dirName, '_');

				peopleOut << dirName << std::endl;

				std::size_t fileCnt    = std::distance(fs::directory_iterator(entry), fs::directory_iterator{});
				std::size_t curFileIdx = 0;

				cv::Mat concat;

				for (const fs::directory_entry& file : fs::directory_iterator(entry))
				{
					curFileIdx++;
					if (fs::is_directory(file)) continue;
					std::cout << "\r[" << args[1] << "] Processing File " << curFileIdx << "/" << fileCnt << std::flush;
					std::vector<cv::Mat> alignedFaces;
					cv::Mat image = cv::imread(file.path().string());

					if (image.size() == cv::Size(FACE_WIDTH, FACE_HEIGHT) && image.channels() == FACE_CHANNEL)
						alignedFaces.push_back(image);
					else
					{
						int32_t width                = image.size().width;
						int32_t height               = image.size().height;
						TrackingObjects trackingDets = detector.Detect(image, 0.5f);

						for (const TrackingObject& box : trackingDets)
						{
							// Crop out only the face
							alignedFaces.push_back(CropAndAlignFace(image, box.bBox));
						}
					}

					for (const cv::Mat& face : alignedFaces)
					{
						cv::Mat pHash;
						std::string hash = "";
						cv::img_hash::pHash(face, pHash);
						for (int32_t i = 0; i < 8; i++)
							hash.append(CharToHexString(pHash.data[i]));

						cv::Mat rep = fr.recognize(face);
						faces[hash] = Face{
							rep, std::stoi(args[0])
						};

						if (concat.empty())
							concat = rep;
						else
							cv::vconcat(concat, rep, concat);
					}
				}

				cv::FileStorage file(dirName + ".yml", cv::FileStorage::WRITE);
				file << args[1] << concat;
				dirTimer.Stop();

				std::cout << " (" << dirTimer << ")" << std::endl;
			}
		}

		peopleOut.close();

		cv::Mat trainData;
		std::vector<int32_t> label;

		for (const auto& [k, v] : faces)
		{
			if (trainData.empty())
				trainData = v.rep;
			else
				cv::vconcat(trainData, v.rep, trainData);

			label.push_back(v.identity);
		}

		std::cout << "Initializing SVM ... " << std::flush;
		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
		svm->setType(cv::ml::SVM::C_SVC);
		svm->setKernel(cv::ml::SVM::LINEAR);
		std::cout << "Done" << std::endl;
		std::cout << "Training SVM ... " << std::flush;
		svm->train(trainData, cv::ml::ROW_SAMPLE, label);
		std::cout << "Done" << std::endl;

		std::cout << "Saving SVM to Disk ... " << std::flush;
		svm->save(SVM_MODEL);
		std::cout << "Done" << std::endl;
		fullTimer.Stop();

		std::cout << "Training SVM done after: " << fullTimer << std::endl;
	}

	void SetVerbose(const bool& en = true)
	{
		m_verbose = en;
	}

	static cv::Mat CropAndAlignFace(const cv::Mat& image, const BBox& bBox)
	{
		// Crop out only the face
		cv::Rect roi(bBox.x, bBox.y, bBox.width - bBox.x, bBox.height - bBox.y);
		cv::Mat crop = image(roi);
		cv::Mat alignedFace;
		cv::resize(crop, alignedFace, cv::Size(FACE_WIDTH, FACE_HEIGHT), 0.0, 0.0, cv::INTER_AREA);

		return alignedFace;
	}

	// ==== Private Static Methods ====
private:
	static void convertToMl2(const std::vector<cv::Mat>& trainSamples, cv::Mat& trainData)
	{
		trainData = cv::Mat();
		for (const cv::Mat& sample : trainSamples)
		{
		}
	}

	// ==== Private Methods ====
private:
	void loadPeople()
	{
		std::ifstream in(PEOPLE_FILE);
		if (!in.is_open())
			throw(FaceRecognitionException(string_format("Unable to open people file: %s", PEOPLE_FILE)));

		m_people.clear();

		std::string str;
		while (in >> str)
		{
			std::vector<std::string> args = splitString(str, '_');
			m_people[std::stoi(args[0])]  = args[1];
		}

		in.close();
	}

	cv::Mat recognize(const cv::Mat& image)
	{
		cv::Mat temp = image.reshape(1, image.rows * 3);
		cv::Mat mean3;
		cv::Mat stdDev3;
		cv::meanStdDev(temp, mean3, stdDev3);

		double meanPxl   = mean3.at<double>(0);
		double stdDevPxl = stdDev3.at<double>(0);
		cv::Mat image2;
		image.convertTo(image2, CV_64FC1);
		image2 = image2 - cv::Scalar(meanPxl, meanPxl, meanPxl);
		image2 = image2 / stdDevPxl;

		tensorflow::Tensor inputTensor = LoadTensorFromImage<float>(image2, tensorflow::DT_FLOAT, CV_32FC3);
		tensorflow::Tensor phaseTensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
		phaseTensor.scalar<bool>()() = 0;

		std::vector<tensorflow::Tensor> outputs;
		std::vector<std::pair<std::string, tensorflow::Tensor>> feedDict = {
			{ INPUT_LAYER, inputTensor } /*,
			{ PHASE_TRAIN_LAYER, phaseTensor }*/
		};

		tensorflow::Status status = m_session->Run(feedDict, OUTPUT_LAYERS, {}, &outputs);
		if (!status.ok()) throw(FaceRecognitionException("Running model failed: " + status.ToString()));

		float* ptr = outputs[0].flat<float>().data();
		cv::Mat res(cv::Size(128, 1), CV_32F, ptr, cv::Mat::AUTO_STEP);

		return res.clone(); // Probably due to memory scoping it is not possible to return the mat itself
	}

private:
	cv::Ptr<cv::ml::SVM> m_svm;
	std::map<uint32_t, std::string> m_people;
	bool m_verbose;
};
