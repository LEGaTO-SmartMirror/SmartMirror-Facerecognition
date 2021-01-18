/* 
 *  File: main.cpp
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

// Disable Tensorflow log messages
// export TF_CPP_MIN_LOG_LEVEL=2

#define DIFF_THRESHOLD 0.005

#include <opencv2/video/video.hpp>

#include "FaceDetector.h"
#include "FaceRecognition.h"
#include "FaceRecognitionRT.h"
#include "SORT.h"
#include "Timer.h"
#include "Types.h"

#include <fstream>
#include <iostream>
#include <thread>

// #define TRAIN_SVM

using namespace std::chrono_literals;

// =========================================================

const int32_t FACE_RECHECK_FRAMES = 15; // Recheck faces in 15 frames again

const double ONE_SECOND = 1000.0;

const uint32_t IMAGE_HEIGHT         = 1920;
const uint32_t IMAGE_WIDTH          = 1080;
const std::string IMAGE_STREAM_PATH = "/dev/shm/camera_1m";
const std::string CAP_STR           = string_format("shmsrc socket-path=%s ! video/x-raw, format=BGR, height=%d, width=%d, framerate=30/1 ! queue max-size-buffers=5 leaky=2 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true", IMAGE_STREAM_PATH.c_str(), IMAGE_HEIGHT, IMAGE_WIDTH);

std::map<uint32_t, TrackingObject> g_boxes;
bool g_stop;
bool g_frameDone = false;
bool g_verbose   = true;

void ProcessFaces()
{
	FaceRecognitionRT fr;
	Timer checkTimer;
	while (!g_stop)
	{
		g_frameDone = false;
		if (!g_boxes.empty())
		{
			for (auto& [id, box] : g_boxes)
			{
				if (box.lastCheck == 0)
				{
					if (g_verbose)
						std::cerr << "Reidentifying Face [" << box.trackingID << "] ... " << std::flush;

					checkTimer.Start();

					uint32_t newFaceID = fr.IdentifyFaceID(box.face);

					if (box.faceID == newFaceID)
					{
						if (box.confidence < 1.0f)
							box.confidence += 0.1f;
					}
					else
					{
						if (box.confidence > 0.0f)
							box.confidence -= 0.1f;
						if (box.confidence < 0.0f)
							box.confidence = 0.0f;
					}

					if (box.confidence <= 0.4f)
					{
						box.faceID = newFaceID;
						box.name   = fr.FaceID2Name(box.faceID);
					}
					box.lastCheck = FACE_RECHECK_FRAMES;
					checkTimer.Stop();

					if (g_verbose)
						std::cerr << "Done after " << checkTimer << " found: " << box.name << std::endl;

					g_frameDone = true;
					break;
				}
			}
		}

		std::this_thread::sleep_for(1ms);
	}
}

void UpdateGlobalBox(const TrackingObjects& boxes, const cv::Mat& frame)
{
	for (const TrackingObject& box : boxes)
	{
		g_boxes.try_emplace(box.trackingID, box);
		g_boxes[box.trackingID].bBox       = box.bBox;
		g_boxes[box.trackingID].lastUpdate = 0;

		// lastCheck == -1 is the initial state so this needs to be handled separately
		if (g_boxes[box.trackingID].lastCheck == -1)
		{
			g_boxes[box.trackingID].face      = FaceRecognitionRT::CropAndAlignFaceFactor(frame, box.bBox);
			g_boxes[box.trackingID].lastCheck = 0;
		}

		if (g_boxes[box.trackingID].lastCheck > 0)
		{
			if (g_boxes[box.trackingID].lastCheck == 1)
				g_boxes[box.trackingID].face = FaceRecognitionRT::CropAndAlignFaceFactor(frame, box.bBox);

			g_boxes[box.trackingID].lastCheck--;
		}
	}

	map_erase_if(g_boxes, [](const std::pair<uint32_t, TrackingObject>& p) { return !p.second.Valid(); });
}

BBox ToCenter(const BBox& bBox)
{
	// x_y = center
	float h = bBox.height - bBox.y;
	float w = bBox.width - bBox.x;
	float x = bBox.x + (w / 2);
	float y = bBox.y + (h / 2);
	return BBox(x, y, w, h);
}

int main(int argc, char* argv[])
{
	try
	{
		std::string image = "modules/SmartMirror-Facerecognition/FaceRecognition_CMake/data/grace_hopper.jpg";
		cv::Mat imageMat;

#ifdef TRAIN_SVM
		FaceRecognitionRT::LearnFromPics();
		return 0;
#endif

		FaceDetector fd;

		float maxFPS        = 30.0f;
		double minFrameTime = 1000.0 / maxFPS;
		TrackingObjects lastObjs;

		//  ============ Preinitialize Tensorflow Models ============
		TrackingObjects boxes;
		cv::Mat frame        = cv::imread(image);
		std::size_t frameCnt = 0;
		std::thread faceRec  = std::thread(ProcessFaces);
		Timer fpsTimer;

		std::cout << "Preinitializing Tensorflow Models ... " << std::flush;
		fpsTimer.Start();
		g_verbose = false;
		boxes     = fd.FindFacesTracked(frame);
		UpdateGlobalBox(boxes, frame);

		while (!g_frameDone)
			std::this_thread::sleep_for(1ms);

		g_boxes.clear();
		fd.ResetTracker();

		fpsTimer.Stop();

		std::cout << "Done after: " << fpsTimer << std::endl;
		std::cout << "CPU Memory usage after initialization: " << GetMemString() << std::endl;
		g_verbose = false;
		//  ============ Preinitialize Tensorflow Models ============

		std::this_thread::sleep_for(std::chrono::seconds(5));

		cv::VideoCapture cap;
		cap.open(CAP_STR);
		if (!cap.isOpened())
		{
			std::cerr << "Unable to open video stream: " << CAP_STR << std::endl;
			return -1;
		}

		std::stringstream str("");
		double elapsedTime;
		double itrTime;
		double fps;

		fd_set readfds;
		FD_ZERO(&readfds);

		// If detection is to fast we need a break
		struct timeval timeout;
		timeout.tv_sec  = 0;
		timeout.tv_usec = 0;
		char message[BUFSIZ];

		fpsTimer.Start();

		while (true)
		{
			bool changed = false;

			FD_SET(STDIN_FILENO, &readfds);
			// check stdin for a new maxFPS amount
			if (select(1, &readfds, NULL, NULL, &timeout))
			{
				if (scanf("%s", message) != EOF)
				{
					float oldFPS = maxFPS;
					maxFPS       = atof(message);
					if (oldFPS != maxFPS)
					{
						std::cout << string_format("{\"STATUS\": \"change FPS to %.1f\"}", maxFPS) << std::endl
								  << std::flush;
					}

					minFrameTime = 1000.0 / maxFPS;
				}
			}

			// if no FPS are needed and maxFPS equals 0, wait and start from the beginning
			if (maxFPS == 0)
			{
				g_boxes.clear();
				lastObjs.clear();
				fd.ResetTracker();
				std::cout << "{\"FACE_DET_FPS\": 0.0}" << std::endl
						  << std::flush;
				std::cout << "{\"DETECTED_FACES\": []}" << std::endl
						  << std::flush;
				std::this_thread::sleep_for(std::chrono::seconds(1));
				continue;
			}

			if (!cap.read(frame))
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				continue;
			}

			str.str("");

			boxes = fd.FindFacesTracked(frame);
			UpdateGlobalBox(boxes, frame);

			if (g_boxes.size() != lastObjs.size())
			{
				changed = true;
				// std::cout << "Size-Change - " << g_boxes.size() << " | " << lastObjs.size() << std::endl;
			}
			else
			{
				for (const auto& [idx, box] : enumerate(g_boxes))
				{
					if (!lastObjs.at(idx).CmpNameAndXY(box.second))
					{
						changed = true;
						// std::cout << "XY-Change" << std::endl;
					}

					if (lastObjs.at(idx).confidence != box.second.confidence)
					{
						changed = true;
						// std::cout << "Conf-Change" << std::endl;
					}
				}
			}

			if (changed)
			{
				lastObjs.clear();
				str << "{\"DETECTED_FACES\": [";
			}

			uint32_t i = 0;

			// Add bounding boxes to frame
			for (auto& [id, box] : g_boxes)
			{
				if (box.Valid() && changed)
				{
					BBox centerBox = ToCenter(box.bBox);
					if (i > 0) str << ", ";

					str << string_format("{\"TrackID\": %d, \"center\": [%.5f,%.5f], \"name\": \"%s\", \"w_h\": [%.5f,%.5f], \"confidence\": %.2f, \"id\": %d}",
										 id, centerBox.x, centerBox.y, box.name.c_str(), centerBox.width, centerBox.height, box.confidence, box.faceID);
					i++;

					lastObjs.push_back(box);
				}

				box.lastUpdate++;
			}

			if (changed)
			{
				str << "]}";
				std::cout << str.str() << std::endl
						  << std::flush;
			}

			frameCnt++;

			fpsTimer.Stop();

			itrTime = fpsTimer.GetElapsedTimeInMilliSec();

			if (itrTime < minFrameTime)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int32_t>(std::round(minFrameTime - itrTime))));
				elapsedTime += minFrameTime;
			}
			else
				elapsedTime += itrTime;

			fps = 1000 / (elapsedTime / frameCnt);

			if (elapsedTime >= ONE_SECOND)
			{
				fps *= 1.2f;
				if (fps > maxFPS) fps = maxFPS;

				std::cout << string_format("{\"FACE_DET_FPS\": %.2f}", fps) << std::endl
						  << std::flush;

				elapsedTime = 0;
				frameCnt    = 0;
			}

			fpsTimer.Start();
		}

		fpsTimer.Stop();

		g_stop = true;

		faceRec.join();
	}
	catch (std::exception& e)
	{
		std::cerr << "EXCEPTION" << std::flush << std::endl;
		std::cerr << e.what() << std::flush << std::endl;
		return -1;
	}

	std::cerr << "FACE RECOGNITION EXITING ... THIS SHOULD NOT HAPPEN" << std::flush << std::endl;
	return 0;
}
