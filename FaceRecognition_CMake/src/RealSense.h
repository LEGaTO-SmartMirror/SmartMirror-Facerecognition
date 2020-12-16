/* 
 *  File: RealSense.h
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

// Intel Realsense Headers
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

// OpenCV Headers
#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#define CONST_VOID_TO_TP(T, V) static_cast<T>(const_cast<void*>(V))

#include <iostream>
#include <thread>

#include "Timer.h"
#include "Utils.h"

class Rotation
{
public:
	enum Direction
	{
		NONE,
		LEFT,
		RIGHT,
		DOUBLE_RIGHT
	};

public:
	Rotation(const Direction& direction = NONE, const uint32_t& width = 0, const uint32_t& height = 0) :
		m_width(width),
		m_height(height),
		m_shiftX(0),
		m_shiftY(0),
		m_angle(0.0),
		m_hasRotation(true)
	{
		switch (direction)
		{
			case NONE:
				m_hasRotation = false;
				return;
			case LEFT:
				m_width  = height;
				m_height = width;
				m_shiftX = height;
				m_shiftY = 0;
				m_angle  = 270.0;
				return;
			case RIGHT:
				m_width  = height;
				m_height = width;
				m_shiftX = 0;
				m_shiftY = width;
				m_angle  = 90.0;
				return;
			case DOUBLE_RIGHT:
				m_shiftX = width;
				m_shiftY = height;
				m_angle  = 180.0;
				return;
		}
	}

	const bool& HasRotation() const
	{
		return m_hasRotation;
	}

	const int32_t& GetWidth() const
	{
		return m_width;
	}

	const int32_t& GetHeight() const
	{
		return m_height;
	}

	const uint32_t& GetShiftX() const
	{
		return m_shiftX;
	}

	const uint32_t& GetShiftY() const
	{
		return m_shiftY;
	}

	const double& GetAngle() const
	{
		return m_angle;
	}

	double GetAngleInRad() const
	{
		return (m_angle * M_PI) / 180;
	}

private:
	int32_t m_width;
	int32_t m_height;
	uint32_t m_shiftX;
	uint32_t m_shiftY;
	double m_angle;
	bool m_hasRotation;
};

class RealSense
{
public:
	static constexpr uint32_t FRAME_RATE         = 30;
	static constexpr uint32_t COLOR_INPUT_WIDTH  = 1280;
	static constexpr uint32_t COLOR_INPUT_HEIGHT = 720;

public:
	RealSense(const std::string& viewName, const std::string& cameraID, const Rotation::Direction& direction = Rotation::NONE) :
		m_pipe(),
		m_cfg(),
		m_viewName(viewName),
		m_cameraID(cameraID),
		m_rotation(direction, COLOR_INPUT_WIDTH, COLOR_INPUT_HEIGHT),
		m_rgbCUDA(),
		m_rgbRotatedCUDA(),
		m_rotated(),
		m_execThread(),
		m_mat(),
		m_run(false)
	{
	}

	void Start()
	{
		m_execThread = std::thread(&RealSense::run, this);
	}

	void Stop()
	{
		m_run = false;
		m_execThread.join();
	}

	const cv::Mat& GetMat() const
	{
		return m_mat;
	}

	bool IsMatValid() const
	{
		return !m_mat.empty();
	}

	const std::string& GetViewName() const
	{
		return m_viewName;
	}

	const Rotation& GetRotation() const
	{
		return m_rotation;
	}

	const cv::Mat& GetRotatedMat()
	{
		m_rgbCUDA.upload(m_mat);

		cv::Size size = { m_rotation.GetWidth(), m_rotation.GetHeight() };
		cv::cuda::rotate(m_rgbCUDA, m_rgbRotatedCUDA, size, m_rotation.GetAngle(), m_rotation.GetShiftX(), m_rotation.GetShiftY());

		m_rgbRotatedCUDA.download(m_rotated);
		return m_rotated;
	}

private:
	void run()
	{
		Timer timer;

		//======================
		// Stream configuration
		//======================
		m_cfg.enable_stream(RS2_STREAM_COLOR, COLOR_INPUT_WIDTH, COLOR_INPUT_HEIGHT, RS2_FORMAT_BGR8, FRAME_RATE);

		m_cfg.enable_device(m_cameraID);

		rs2::pipeline_profile profile = m_pipe.start(m_cfg);
		rs2::device selected_device   = profile.get_device();

		// Wait for frames from the camera to settle
		for (uint32_t i = 0; i < 30; i++)
			m_pipe.wait_for_frames(); // Drop several frames for auto-exposure

		// Capture a single frame and obtain depth + RGB values from it
		rs2::frameset frames   = m_pipe.wait_for_frames();
		rs2::video_frame color = frames.get_color_frame();

		m_run = true;

#if LIMIT_FPS
		std::chrono::high_resolution_clock::time_point before = std::chrono::high_resolution_clock::now();
#endif

		while (m_run)
		{
#if LIMIT_FPS
			before = std::chrono::high_resolution_clock::now();
#endif

			timer.Start();

			// Capture a single frame and obtain depth + RGB values from it
			frames = m_pipe.wait_for_frames();
			color  = frames.get_color_frame();

			m_mat = cv::Mat(COLOR_INPUT_HEIGHT, COLOR_INPUT_WIDTH, CV_8UC3, CONST_VOID_TO_TP(uint8_t*, color.get_data()));

#if LIMIT_FPS
			// Actifly limit framerate to 30 FPS
			std::this_thread::sleep_until<std::chrono::system_clock>(before + std::chrono::milliseconds(1000 / (FRAME_RATE + 1)));
#endif

			timer.Stop();
			// std::cout << m_viewName << ": Got Frame after: " << timer.GetElapsedTimeInMilliSec() << " ms - " << 1000 / timer.GetElapsedTimeInMilliSec() << " FPS" << std::endl;
		}

		std::cout << "Stopping Thread: " << m_viewName << std::endl;
	}

private:
	// RealSense pipeline, encapsulating the actual device and sensors
	rs2::pipeline m_pipe;

	// Configuration for the pipeline with a non default profile
	rs2::config m_cfg;

	std::string m_viewName;
	std::string m_cameraID;
	Rotation m_rotation;

	cv::cuda::GpuMat m_rgbCUDA;
	cv::cuda::GpuMat m_rgbRotatedCUDA;
	cv::Mat m_rotated;

	std::thread m_execThread;
	cv::Mat m_mat;
	bool m_run;
};
