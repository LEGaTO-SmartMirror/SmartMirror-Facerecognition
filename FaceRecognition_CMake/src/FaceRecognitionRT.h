/* 
 *  File: FaceRecognitionRT.h
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
#include "Timer.h"
#include "Types.h"
#include "Utils.h"

// == TensorRT includes ==
#include "argsParser.h"
#include "buffers.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include <NvInfer.h>
// == TensorRT includes ==

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/img_hash/phash.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

DEFINE_EXCEPTION(FaceRecognitionRTException);

class FaceRecognitionRT
{
	struct Face
	{
		cv::Mat rep;
		int32_t identity;
	};

	inline static const std::string ONNX_FILE    = "modules/SmartMirror-Facerecognition/FaceRecognition_CMake/model/facenet_keras.onnx";
	inline static const std::string ENGINE_FILE  = "modules/SmartMirror-Facerecognition/FaceRecognition_CMake/model/FaceNet.engine";
	inline static const std::string SVM_MODEL    = "modules/SmartMirror-Facerecognition/FaceRecognition_CMake/model/svm_trt.yaml";
	inline static const std::string PEOPLE_FILE  = "modules/SmartMirror-Facerecognition/FaceRecognition_CMake/model/people.txt";
	inline static const std::string INPUT_LAYER  = "input_1:0";
	inline static const std::string OUTPUT_LAYER = "Bottleneck_BatchNorm/batchnorm_1/add_1:0";

	inline static const std::string TRAINING_DATA_FOLDER = "modules/SmartMirror-Facerecognition/FaceRecognition_CMake/facedatabase/images";
	inline static const uint32_t FACE_WIDTH              = 160;
	inline static const uint32_t FACE_HEIGHT             = 160;
	inline static const uint32_t FACE_CHANNEL            = 3;

	inline static const bool USE_FP16    = true;
	inline static const int32_t DLA_CORE = 0; // -1 => Do not use DLAs

	template<typename T>
	using InferUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
	FaceRecognitionRT(const bool& trainMode = false) :
		m_engine(nullptr),
		m_context(nullptr),
		m_svm(),
		m_people(),
		m_verbose(false)
	{
		TrtLog::gLogger.setReportableSeverity(TrtLog::Severity::kWARNING);

		buildOrLoadEngine();

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

	uint32_t IdentifyFaceID(const cv::Mat& face, const bool& verbose = false)
	{
		if (face.size() != cv::Size(FACE_WIDTH, FACE_HEIGHT)) return -1;

		cv::Mat rep = recognize(face, verbose);

		return static_cast<uint32_t>(m_svm->predict(rep));
	}

	std::string FaceID2Name(const uint32_t id) const
	{
		if (m_people.count(id) < 1) return "Unknown - Not in List";

		return m_people.at(id);
	}

	std::string IdentifyFace(const cv::Mat& face, const bool& verbose = false)
	{
		uint32_t id = IdentifyFaceID(face, verbose);
		return FaceID2Name(id);
	}

	static void LearnFromPics()
	{
		FaceDetector detector;
		FaceRecognitionRT fr(true);
		std::map<std::string, Face> faces;
		std::ofstream peopleOut(PEOPLE_FILE);
		Timer fullTimer;
		Timer dirTimer;

		if (!peopleOut.is_open())
			throw(FaceRecognitionRTException(string_format("Unable to create people file: %s", PEOPLE_FILE)));

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
							alignedFaces.push_back(CropAndAlignFaceFactor(image, box.bBox));
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

	static cv::Mat CropAndAlignFace(const cv::Mat& image, const cv::Rect& bBox)
	{
		int32_t x = bBox.x;
		int32_t y = bBox.y;
		if (x < 0) x = 0;
		if (y < 0) y = 0;

		int32_t w = std::abs(bBox.width - bBox.x);
		int32_t h = std::abs(bBox.height - bBox.y);

		if (w < 0) w = 0;
		if (h < 0) h = 0;
		if (w > image.cols - 1) w = image.cols - 1;
		if (h > image.rows - 1) h = image.rows - 1;

		// Crop out only the face
		cv::Rect roi(x, y, w, h);
		cv::Mat crop = image(roi);
		cv::Mat alignedFace;
		cv::resize(crop, alignedFace, cv::Size(FACE_WIDTH, FACE_HEIGHT), 0.0, 0.0, cv::INTER_AREA);

		return alignedFace;
	}

	static cv::Mat CropAndAlignFaceFactor(const cv::Mat& image, const BBox& bBox)
	{
		int32_t imgW = image.size().width;
		int32_t imgH = image.size().height;

		int32_t x = bBox.x * imgW;
		int32_t y = bBox.y * imgH;
		int32_t w = bBox.width * imgW;
		int32_t h = bBox.height * imgH;

		if (x < 0) x = 0;
		if (y < 0) y = 0;
		if (w > imgW - 1) w = imgW - 1;
		if (h > imgH - 1) h = imgH - 1;

		return CropAndAlignFace(image, cv::Rect(x, y, w, h));
	}

private:
	void buildOrLoadEngine()
	{
		if (fileExists(ENGINE_FILE))
			loadEngine();
		else
			buildEngine();
	}

	void loadEngine()
	{
		std::cout << "Loading serialized engine ... " << std::flush;

		std::ifstream file(ENGINE_FILE, std::ios::binary);
		if (!file) throw(FaceRecognitionRTException(string_format("[LoadEngine] Failed to open Engine File: %s", ENGINE_FILE)));

		file.seekg(0, file.end);
		std::size_t size = file.tellg();
		file.seekg(0, file.beg);

		std::vector<char> engineData(size);
		file.read(engineData.data(), size);
		file.close();

		InferUniquePtr<nvinfer1::IRuntime> pRuntime{ nvinfer1::createInferRuntime(TrtLog::gLogger.getTRTLogger()) };
		if (!pRuntime) throw(FaceRecognitionRTException("Failed to create InferRuntime"));

		if (DLA_CORE >= 0)
		{
			std::cout << " - Enabling DLACore=" << DLA_CORE << " - " << std::flush;
			pRuntime->setDLACore(DLA_CORE);
		}

		m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(pRuntime->deserializeCudaEngine(engineData.data(), size, nullptr), samplesCommon::InferDeleter());
		if (!m_engine) throw(FaceRecognitionRTException("Failed to deserialize Engine"));

		m_context = InferUniquePtr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
		if (!m_context) throw(FaceRecognitionRTException("Failed to create Execution Context"));

		std::cout << "Done" << std::endl;
	}

	void buildEngine()
	{
		auto builder = InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(TrtLog::gLogger.getTRTLogger()));
		if (!builder) throw(FaceRecognitionRTException("Failed to create Builder"));

		const uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

		auto network = InferUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
		if (!network) throw(FaceRecognitionRTException("Failed to create Network"));

		auto config = InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
		if (!config) throw(FaceRecognitionRTException("Failed to create BuilderConfig"));

		auto parser = InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, TrtLog::gLogger.getTRTLogger()));
		if (!parser) throw(FaceRecognitionRTException("Failed to create Parser"));

		if (!parser->parseFromFile(ONNX_FILE.c_str(), static_cast<int>(TrtLog::gLogger.getReportableSeverity())))
			throw(FaceRecognitionRTException("Failed to parse ONNX File"));

		config->setMaxWorkspaceSize(((size_t)1) << 33);
		config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

		if (USE_FP16)
			config->setFlag(nvinfer1::BuilderFlag::kFP16);

		samplesCommon::enableDLA(builder.get(), config.get(), DLA_CORE);

		m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
		if (!m_engine) throw(FaceRecognitionRTException("Failed to create Build Engine"));

		m_context = InferUniquePtr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
		if (!m_context) throw(FaceRecognitionRTException("Failed to create Execution Context"));

		std::cout << "Writing engine file to disk ... " << std::flush;
		std::ofstream engineFile(ENGINE_FILE, std::ios::binary);
		if (!engineFile) throw(FaceRecognitionRTException(string_format("[SaveEngine] Failed to open Engine File: %s", ENGINE_FILE)));

		InferUniquePtr<nvinfer1::IHostMemory> pSerializedEngine{ m_engine->serialize() };
		if (!pSerializedEngine) throw(FaceRecognitionRTException("Failed to serialize Engine"));

		engineFile.write(static_cast<char*>(pSerializedEngine->data()), pSerializedEngine->size());
		engineFile.close();

		std::cout << "Done" << std::endl;
	}

	void loadPeople()
	{
		std::ifstream in(PEOPLE_FILE);
		if (!in.is_open())
			throw(FaceRecognitionRTException(string_format("Unable to open people file: %s", PEOPLE_FILE)));

		m_people.clear();

		std::string str;
		while (in >> str)
		{
			std::vector<std::string> args = splitString(str, '_');
			m_people[std::stoi(args[0])]  = args[1];
		}

		in.close();
	}

	cv::Mat recognize(const cv::Mat& image, const bool& verbose = false)
	{
		// Create RAII buffer manager object
		samplesCommon::BufferManager buffers(m_engine);

		Timer timer;
		timer.Start();

		// ==== Inference ====

		processInput(buffers, image);

		// Copy data from host input buffers to device input buffers
		buffers.copyInputToDevice();

		// Execute the inference work
		if (!m_context->executeV2(buffers.getDeviceBindings().data()))
			throw(FaceRecognitionRTException("Inference execution failed"));

		// Copy data from device output buffers to host output buffers
		buffers.copyOutputToHost();

		// ==== Inference ====

		timer.Stop();
		// std::cout << "Inference-Timer: " << timer << std::endl;

		float* ptr = static_cast<float*>(buffers.getHostBuffer(OUTPUT_LAYER));
		cv::Mat res(cv::Size(128, 1), CV_32F, ptr, cv::Mat::AUTO_STEP);

		if (verbose)
		{
			std::cout << " === Inference Results ===" << std::endl;
			for (uint32_t i = 0; i < 128; i++)
				std::cout << ptr[i] << std::endl;
			std::cout << " === Inference Results ===" << std::endl;
		}

		return res.clone(); // Probably due to memory scoping it is not possible to return the mat itself
	}

	void processInput(const samplesCommon::BufferManager& buffers, const cv::Mat& image) const
	{
		// ==== Pre-Process Image ====
		cv::Mat temp = image.reshape(1, image.rows * 3);
		cv::Mat mean3;
		cv::Mat stdDev3;
		cv::meanStdDev(temp, mean3, stdDev3);

		double meanPxl   = mean3.at<double>(0);
		double stdDevPxl = stdDev3.at<double>(0);
		cv::Mat image2;
		cv::Mat image3;
		image.convertTo(image2, CV_64FC1);
		image2 = image2 - cv::Scalar(meanPxl, meanPxl, meanPxl);
		image2 = image2 / stdDevPxl;

		image2.convertTo(image3, CV_32FC3);
		image2 = image3;
		// ==== Pre-Process Image ====
		cv::Size imageSize = image.size();

		float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(INPUT_LAYER));
		std::memcpy(hostInputBuffer, image2.ptr<float>(0), imageSize.height * imageSize.width * image2.channels() * sizeof(float));
	}

	bool verifyOutput(const samplesCommon::BufferManager& buffers, const std::string& outputTensorName) const
	{
		const float* prob = static_cast<const float*>(buffers.getHostBuffer(outputTensorName));

		// Print output values for each index
		for (int j = 0; j < 128; j++)
			std::cout << prob[j] << std::endl;

		return true;
	}

	static bool fileExists(const std::string& name)
	{
		std::ifstream f(name);
		return f.good();
	}

private:
	std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
	InferUniquePtr<nvinfer1::IExecutionContext> m_context;

	cv::Ptr<cv::ml::SVM> m_svm;
	std::map<uint32_t, std::string> m_people;
	bool m_verbose;
};
