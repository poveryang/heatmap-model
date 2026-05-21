#ifndef HMAP_GENERATOR_HMAP_GENERATOR_H
#define HMAP_GENERATOR_HMAP_GENERATOR_H


#include "c_api.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


struct Hotspot {
    int id=-1;                                      // The id of the hotspot
    int type=-1;                                    // The type of the barcode in the hotspot
    int area=0;                                     // The area of the hotspot(note: the area isn't same as the rect)
    double sum_intensity=0.;                        // The sum intensity of the hotspot
    double mean_intensity=0.;                       // The mean intensity of the hotspot
    double max_intensity=0.;                        // The max intensity of the hotspot
    cv::Rect2d rect={0,0,0,0};    // The rectangle of the hotspot
    cv::Mat mask=cv::Mat();                         // The mask of the hotspot inside the rect
    cv::Point2d centroid={0,0};               // The centroid of the hotspot
};

class HeatMapGenerator {
public:
    explicit HeatMapGenerator(const std::string &context_name="timvx", const std::string &precision="uint8");

    ~HeatMapGenerator();

    void Init(const std::string &model_path);

    cv::Mat Infer(const cv::Mat &image);

    std::vector<Hotspot> LocateHotspots(const cv::Mat &heatmap, double intensity_thres=40);

    Hotspot LocateMaxHotspot(const cv::Mat &heatmap, double intensity_thres=40);

public:
    std::vector<Hotspot> hots_recoder;

private:
    cv::Mat InferFP32(cv::Mat &image);

    cv::Mat InferUInt8(cv::Mat &image);

    cv::Mat PreProcess(const cv::Mat &image);

    cv::Mat PostProcess(const cv::Mat &image, cv::Size &dst_size);

    static cv::Mat Sigmoid(const cv::Mat &image);

    cv::Mat OutputTensorToMat(const void *data, int depth) const;

    void IdentifyHotspot(Hotspot &hotspot);

    static double CalcHotsIOU(const Hotspot &hot1, const Hotspot &hot2);

private:
    // quantization parameters
    float input_scale = 0.f;
    int input_zero_point = 0;
    float output_scale = 0.f;
    int output_zero_point = 0;

    // model parameters
    graph_t graph_{};
    options opt_{};
    tensor_t input_tensor_{};
    tensor_t output_tensor_{};
    int input_buffer_size_{};
    int out_dim_[4]{};
    std::string context_name_;
    std::string precision_;
};


#endif //HMAP_GENERATOR_HMAP_GENERATOR_H
