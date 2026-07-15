#ifndef HMAP_YOLO_DETECTOR_H
#define HMAP_YOLO_DETECTOR_H

#include "c_api.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct YoloDetection {
    cv::Rect2f box;
    float score = 0.f;
    int class_id = -1;
};

struct YoloTiming {
    double preprocess_ms = 0.;
    double inference_ms = 0.;
    double postprocess_ms = 0.;
    double total_ms = 0.;
};

class YoloDetector {
public:
    explicit YoloDetector(const std::string& context_name = "timvx",
                          const std::string& precision = "uint8");
    ~YoloDetector();

    bool Init(const std::string& model_path);
    double Warmup(int count);
    bool Infer(const cv::Mat& image, std::vector<YoloDetection>& detections, YoloTiming& timing);

    static void DrawDetections(cv::Mat& image, const std::vector<YoloDetection>& detections);
    static const char* ClassName(int class_id);

private:
    struct LetterboxInfo {
        float scale = 1.f;
        int left = 0;
        int top = 0;
        int original_width = 0;
        int original_height = 0;
    };

    bool Preprocess(const cv::Mat& image, LetterboxInfo& info);
    bool Decode(const LetterboxInfo& info, std::vector<YoloDetection>& detections) const;
    bool DecodePerLevel(const LetterboxInfo& info,
                        std::vector<YoloDetection>& detections) const;
    static float OutputValue(const void* output, int index, int data_type,
                             float scale, int zero_point);
    static float IoU(const cv::Rect2f& a, const cv::Rect2f& b);
    static void Nms(std::vector<YoloDetection>& detections, float iou_threshold);

    std::string context_name_;
    std::string precision_;
    bool tengine_initialized_ = false;
    bool graph_prerun_ = false;
    context_t context_ = nullptr;
    graph_t graph_ = nullptr;
    tensor_t input_tensor_ = nullptr;
    tensor_t boxes_tensor_ = nullptr;
    tensor_t scores_tensor_ = nullptr;

    struct RawHeadTensor {
        tensor_t tensor = nullptr;
        int channels = 0;
        int height = 0;
        int width = 0;
        int data_type = -1;
        float scale = 1.f;
        int zero_point = 0;
    };

    struct RawHeadLevel {
        int stride = 0;
        RawHeadTensor boxes;
        RawHeadTensor scores;
    };

    std::vector<RawHeadLevel> raw_head_levels_;
    options options_{};

    int input_width_ = 640;
    int input_height_ = 640;
    int input_channels_ = 3;
    int boxes_dims_[4]{};
    int scores_dims_[4]{};
    int boxes_dim_count_ = 0;
    int scores_dim_count_ = 0;
    int boxes_data_type_ = -1;
    int scores_data_type_ = -1;
    int boxes_elements_ = 0;
    int scores_elements_ = 0;

    float input_scale_ = 1.f;
    int input_zero_point_ = 0;
    float boxes_scale_ = 1.f;
    int boxes_zero_point_ = 0;
    float scores_scale_ = 1.f;
    int scores_zero_point_ = 0;
    float confidence_threshold_ = 0.25f;
    float nms_iou_threshold_ = 0.45f;

    std::vector<uint8_t> input_uint8_;
    std::vector<float> input_float_;
};

#endif
