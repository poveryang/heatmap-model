#include "yolo_detector.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " -c context -p precision -m model -d input_dir -o output_dir [-w warmup]"
              << std::endl;
}

bool has_image_extension(const std::string& path) {
    const auto dot = path.rfind('.');
    if (dot == std::string::npos) {
        return false;
    }
    std::string extension = path.substr(dot + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension == "png" || extension == "jpg" || extension == "jpeg" || extension == "bmp";
}

void collect_images(const std::string& directory, std::vector<std::string>& images) {
    DIR* handle = opendir(directory.c_str());
    if (handle == nullptr) {
        return;
    }
    dirent* entry = nullptr;
    while ((entry = readdir(handle)) != nullptr) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        const std::string path = directory + "/" + entry->d_name;
        struct stat state {};
        if (stat(path.c_str(), &state) != 0) {
            continue;
        }
        if (S_ISDIR(state.st_mode)) {
            collect_images(path, images);
        } else if (S_ISREG(state.st_mode) && has_image_extension(path)) {
            images.push_back(path);
        }
    }
    closedir(handle);
}

void mkdir_recursive(const std::string& directory) {
    if (directory.empty()) {
        return;
    }
    std::string current;
    for (size_t i = 0; i < directory.size(); ++i) {
        current.push_back(directory[i]);
        if (directory[i] == '/' && current.size() > 1) {
            mkdir(current.c_str(), 0755);
        }
    }
    mkdir(directory.c_str(), 0755);
}

std::string relative_path(const std::string& root, const std::string& path) {
    if (path.compare(0, root.size(), root) != 0) {
        return path;
    }
    std::string result = path.substr(root.size());
    if (!result.empty() && result.front() == '/') {
        result.erase(result.begin());
    }
    return result;
}

std::string output_path(const std::string& output_root, const std::string& relative) {
    const auto dot = relative.rfind('.');
    const std::string stem = dot == std::string::npos ? relative : relative.substr(0, dot);
    return output_root + "/" + stem + ".det.png";
}

std::string parent_path(const std::string& path) {
    const auto slash = path.rfind('/');
    return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

double average(const std::vector<double>& values) {
    double total = 0.;
    for (const double value : values) {
        total += value;
    }
    return values.empty() ? 0. : total / static_cast<double>(values.size());
}

double percentile(std::vector<double> values, double ratio) {
    if (values.empty()) {
        return 0.;
    }
    std::sort(values.begin(), values.end());
    const double position = ratio * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1. - fraction) + values[upper] * fraction;
}

}  // namespace

int main(int argc, char** argv) {
    std::string context = "timvx";
    std::string precision = "uint8";
    std::string model_path;
    std::string input_dir;
    std::string output_dir;
    int warmup_count = 3;

    int option = 0;
    while ((option = getopt(argc, argv, "c:p:m:d:o:w:")) != -1) {
        switch (option) {
            case 'c': context = optarg; break;
            case 'p': precision = optarg; break;
            case 'm': model_path = optarg; break;
            case 'd': input_dir = optarg; break;
            case 'o': output_dir = optarg; break;
            case 'w': warmup_count = std::max(0, std::atoi(optarg)); break;
            default: print_usage(argv[0]); return 2;
        }
    }
    if (model_path.empty() || input_dir.empty() || output_dir.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    std::vector<std::string> images;
    collect_images(input_dir, images);
    std::sort(images.begin(), images.end());
    if (images.empty()) {
        fprintf(stderr, "No input images under %s\n", input_dir.c_str());
        return 1;
    }
    mkdir_recursive(output_dir);

    YoloDetector detector(context, precision);
    const auto init_begin = Clock::now();
    if (!detector.Init(model_path)) {
        return 1;
    }
    const double init_ms = std::chrono::duration<double, std::milli>(Clock::now() - init_begin).count();
    const char* hold_after_init = std::getenv("YOLO_HOLD_AFTER_INIT_SECONDS");
    const int hold_seconds = hold_after_init == nullptr ? 0 : std::max(0, std::atoi(hold_after_init));
    if (hold_seconds > 0) {
        printf("resident_measurement pid=%d init_ms=%.3f hold_seconds=%d\n",
               static_cast<int>(getpid()), init_ms, hold_seconds);
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(hold_seconds));
    }
    const double warmup_ms = detector.Warmup(warmup_count);
    if (warmup_ms < 0.) {
        return 1;
    }
    printf("benchmark_setup init_ms=%.3f warmup_count=%d warmup_inference_avg_ms=%.3f\n",
           init_ms, warmup_count, warmup_ms);

    std::ofstream csv(output_dir + "/results.csv");
    csv << "image,detections,preprocess_ms,inference_ms,postprocess_ms,total_ms\n";
    const bool write_output_images = std::getenv("YOLO_SKIP_OUTPUT_IMAGES") == nullptr;
    const bool write_detection_details = std::getenv("YOLO_WRITE_DETECTIONS") != nullptr;
    std::ofstream detection_csv;
    if (write_detection_details) {
        detection_csv.open(output_dir + "/detections.csv");
        detection_csv << "image,index,class_id,class_name,score,x0,y0,x1,y1\n";
    }
    printf("output_images=%s\n", write_output_images ? "enabled" : "disabled");
    std::vector<double> preprocess_times;
    std::vector<double> inference_times;
    std::vector<double> postprocess_times;
    std::vector<double> total_times;
    preprocess_times.reserve(images.size());
    inference_times.reserve(images.size());
    postprocess_times.reserve(images.size());
    total_times.reserve(images.size());
    size_t detection_sum = 0;
    int processed = 0;

    for (const std::string& path : images) {
        cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            fprintf(stderr, "Read image failed: %s\n", path.c_str());
            return 1;
        }
        std::vector<YoloDetection> detections;
        YoloTiming timing;
        if (!detector.Infer(image, detections, timing)) {
            fprintf(stderr, "Inference failed: %s\n", path.c_str());
            return 1;
        }

        const std::string relative = relative_path(input_dir, path);
        if (write_detection_details) {
            for (size_t index = 0; index < detections.size(); ++index) {
                const YoloDetection& detection = detections[index];
                detection_csv << relative << ',' << index << ',' << detection.class_id << ','
                              << YoloDetector::ClassName(detection.class_id) << ',' << std::fixed
                              << std::setprecision(6) << detection.score << ',' << detection.box.x
                              << ',' << detection.box.y << ',' << detection.box.br().x << ','
                              << detection.box.br().y << '\n';
            }
        }
        if (write_output_images) {
            const std::string annotated_path = output_path(output_dir, relative);
            mkdir_recursive(parent_path(annotated_path));
            cv::Mat annotated;
            if (image.channels() == 1) {
                cv::cvtColor(image, annotated, cv::COLOR_GRAY2BGR);
            } else if (image.channels() == 4) {
                cv::cvtColor(image, annotated, cv::COLOR_BGRA2BGR);
            } else {
                annotated = image.clone();
            }
            YoloDetector::DrawDetections(annotated, detections);
            if (!cv::imwrite(annotated_path, annotated)) {
                fprintf(stderr, "Write result failed: %s\n", annotated_path.c_str());
                return 1;
            }
        }

        preprocess_times.push_back(timing.preprocess_ms);
        inference_times.push_back(timing.inference_ms);
        postprocess_times.push_back(timing.postprocess_ms);
        total_times.push_back(timing.total_ms);
        detection_sum += detections.size();
        ++processed;

        csv << relative << ',' << detections.size() << ',' << std::fixed << std::setprecision(3)
            << timing.preprocess_ms << ',' << timing.inference_ms << ','
            << timing.postprocess_ms << ',' << timing.total_ms << '\n';
        printf("image_result index=%d/%zu image=%s detections=%zu preprocess_ms=%.3f "
               "inference_ms=%.3f postprocess_ms=%.3f total_ms=%.3f\n",
               processed, images.size(), relative.c_str(), detections.size(), timing.preprocess_ms,
               timing.inference_ms, timing.postprocess_ms, timing.total_ms);
    }

    const double average_total = average(total_times);
    printf("benchmark_summary images=%d detections=%zu "
           "preprocess_avg_ms=%.3f preprocess_p50_ms=%.3f preprocess_p95_ms=%.3f "
           "inference_avg_ms=%.3f inference_p50_ms=%.3f inference_p95_ms=%.3f "
           "postprocess_avg_ms=%.3f postprocess_p50_ms=%.3f postprocess_p95_ms=%.3f "
           "total_avg_ms=%.3f total_p50_ms=%.3f total_p95_ms=%.3f pipeline_fps=%.3f\n",
           processed, detection_sum,
           average(preprocess_times), percentile(preprocess_times, 0.50), percentile(preprocess_times, 0.95),
           average(inference_times), percentile(inference_times, 0.50), percentile(inference_times, 0.95),
           average(postprocess_times), percentile(postprocess_times, 0.50), percentile(postprocess_times, 0.95),
           average_total, percentile(total_times, 0.50), percentile(total_times, 0.95),
           1000. / average_total);
    return 0;
}
