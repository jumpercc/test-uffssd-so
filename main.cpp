#include <iostream>
#include <string>
#include <chrono>
#include "uffssd.h"
#include <unistd.h>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>

std::string gstreamer_pipeline(
        int capture_width, int capture_height,
        int display_width, int display_height,
        int framerate,
        int flip_method
) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) +
           ", height=(int)" + std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" +
           std::to_string(framerate) + "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) +
           " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) +
           ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int main() {
    const int capture_width = 1280;  // 3264;
    const int capture_height = 720;  // 2464;
    const int display_width = 300;
    const int display_height = 300;
    const int framerate = 1;
    const int flip_method = 0;
    const std::string net_files_directory = "/home/ballsbot/projects/test-uffssd-so";

    std::string pipeline = gstreamer_pipeline(
            capture_width,
            capture_height,
            display_width,
            display_height,
            framerate,
            flip_method
    );

    std::vector<std::string> classes(91);
    std::ifstream labelFile(net_files_directory + "/ssd_coco_labels.txt");
    std::string line;
    int id = 0;
    while (getline(labelFile, line)) {
        classes[id++] = line;
    }

    StartupDetector(net_files_directory);

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera." << std::endl;
        return (-1);
    }

    cv::Mat img;
    uint8_t raw_bgr[display_width * display_height * 3];

    while (true) {
        if (!cap.read(img)) {
            std::cerr << "Capture read error" << std::endl;
            break;
        }

        if (img.type() != CV_8UC3) {
            std::cerr << "Unexpected capture format (need CV_8UC3)" << std::endl;
            break;
        }

        size_t offset = 0;
        for (size_t r = 0; r < img.rows; ++r) {
            for (size_t c = 0; c < img.cols; ++c) {
                auto &pixel = img.at<cv::Vec3b>(r, c);
                raw_bgr[offset++] = pixel[2];
                raw_bgr[offset++] = pixel[1];
                raw_bgr[offset++] = pixel[0];
            }
        }

        auto detections = DetectObjects(&raw_bgr[0]);
        std::cout << "detected " << detections.size() << " objects\n";
        for (auto it: detections) {
            std::cout << "class: " << classes[int(it[0])] << ", confidence: " << it[1] * 100.f << " (" << it[2] << ", "
                      << it[3]
                      << ") , (" << it[4] << ", " << it[5] << ")\n";
        }
    }

    cap.release();
    CleanupDetector();
    return 0;
}
