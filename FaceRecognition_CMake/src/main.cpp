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

// #define PRINT_MEMORY_USAGE

#define WEBCAM

using namespace std::chrono_literals;

// =========================================================

const int32_t FACE_RECHECK_FRAMES = 30; // Recheck faces in 30 frames again

const double ONE_SECOND = 1000.0;

const uint32_t IMAGE_HEIGHT         = 1920;
const uint32_t IMAGE_WIDTH          = 1080;
const std::string IMAGE_STREAM_PATH = "/dev/shm/camera_image";

std::map<uint32_t, TrackingObject> g_boxes;
bool g_stop;
bool g_frameDone = false;
bool g_verbose   = true;

std::string g_capStr = string_format("shmsrc socket-path=%s ! video/x-raw, format=BGR, height=%d, width=%d, framerate=30/1 ! queue max-size-buffers=5 leaky=2 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true", IMAGE_STREAM_PATH.c_str(), IMAGE_HEIGHT, IMAGE_WIDTH);

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
					box.faceID    = fr.IdentifyFaceID(box.face);
					box.name      = fr.FaceID2Name(box.faceID);
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
		// TODO: cut out face each time or just every x-Frames?
		g_boxes[box.trackingID].face = FaceRecognitionRT::CropAndAlignFaceFactor(frame, box.bBox);

		if (g_boxes[box.trackingID].lastCheck == -1)
			g_boxes[box.trackingID].lastCheck = 0;

		if (g_boxes[box.trackingID].lastCheck > 0)
			g_boxes[box.trackingID].lastCheck--;
	}
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
	std::string image = "modules/SmartMirror-Facerecognition/FaceRecognition_CMake/data/grace_hopper.jpg";
	cv::Mat imageMat;

#ifdef TRAIN_SVM
	FaceRecognitionRT::LearnFromPics();
	return 0;
#endif

	FaceDetector fd;

	float maxFPS        = 30.0;
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
	cap.open(g_capStr);
	if (!cap.isOpened())
	{
		std::cerr << "Unable to open video stream: " << g_capStr << std::endl;
		return 0;
	}

	std::stringstream str("");
	double elapsedTime;
	double itrTime;
	double fps;

	fd_set readfds;
	FD_ZERO(&readfds);
	/* If detection is to fast we need  a break */
	struct timeval timeout;
	timeout.tv_sec  = 0;
	timeout.tv_usec = 0;
	char message[BUFSIZ];
	uint32_t equalityCounter = 0;

	fpsTimer.Start();

	while (true)
	{
		bool changed = false;
		// check stdin for a new maxFPS amount
		FD_SET(STDIN_FILENO, &readfds);

		if (select(1, &readfds, NULL, NULL, &timeout))
		{
			if (scanf("%s", message) != EOF)
			{
				maxFPS       = atof(message);
				minFrameTime = 1000.0 / maxFPS;
			}
		}

		// if no FPS are needed and maxFPS equals 0, wait and start from the beginning
		if (maxFPS == 0)
		{
			std::this_thread::sleep_for(std::chrono::seconds(1));
			printf("{\"FACE_DET_FPS\": 0.0}\n");
			fflush(stdout);
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

		equalityCounter = 0;
		for (const TrackingObject& lastObj : lastObjs)
		{
			for (const auto& [id, box] : g_boxes)
			{
				if (lastObj.CmpNameAndXY(box))
					equalityCounter++;
			}
		}

		if (g_boxes.size() != lastObjs.size() || g_boxes.size() != equalityCounter)
			changed = true;

		lastObjs.clear();
		lastObjs.reserve(g_boxes.size());
		if (changed) str << "{\"DETECTED_FACES\": [";
		uint32_t i = 0;

		// Add bounding boxes to frame
		for (auto& [id, box] : g_boxes)
		{
			if (box.Valid() && changed)
			{
				BBox centerBox = ToCenter(box.bBox);
				if (i > 0) str << ", ";

				str << string_format("{\"TrackID\": %d, \"center\": [%.5f,%.5f], \"name\": \"%s\", \"w_h\": [%.5f,%.5f], \"confidence\": 1.0, \"id\": %d}",
									 id, centerBox.x, centerBox.y, box.name.c_str(), centerBox.width, centerBox.height, box.faceID);
				i++;
			}

			box.lastUpdate++;
			lastObjs.push_back(box);
		}

		if (changed)
		{
			str << "]}";
			std::cout << str.str() << std::endl;
		}

		frameCnt++;

		fpsTimer.Stop();

		itrTime = fpsTimer.GetElapsedTimeInMilliSec();
		elapsedTime += itrTime;
		fps = 1000 / (elapsedTime / frameCnt);

		if (itrTime < minFrameTime)
			std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int32_t>(std::round(minFrameTime - itrTime))));

		if (elapsedTime >= ONE_SECOND)
		{
			if (fps > maxFPS) fps = maxFPS;

			std::cout << string_format("{\"FACE_DET_FPS\": %.2f}\n", fps) << std::flush;

			elapsedTime = 0;
			frameCnt    = 0;
		}

		fpsTimer.Start();
	}

	fpsTimer.Stop();

	g_stop = true;

	faceRec.join();

	return 0;
}
