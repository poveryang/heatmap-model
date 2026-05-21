#include <iostream>
#include <unistd.h>
#include "hmap_generator.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>


int main(int argc, char **argv) {
    /* Parse arguments */
    int ch;
    std::string context = "cpu";
    std::string precision = "uint8";
    std::string model_path;
    std::string image_path;
    while ((ch = getopt(argc, argv, "c:p:m:i:")) != -1) {
        switch (ch) {
            case 'c':
                context = optarg;
                break;
            case 'p':
                precision = optarg;
                break;
            case 'm':
                model_path = optarg;
                break;
            case 'i':
                image_path = optarg;
                break;
            default:
                std::cout << "Usage: " << argv[0]
                          << " -c context -p precision -m model_path -i image_path" << std::endl;
                exit(1);
        }
    }

    if (model_path.empty() || image_path.empty()) {
        std::cout << "Usage: " << argv[0]
                  << " -c context -p precision -m model_path -i image_path" << std::endl;
        exit(1);
    }

    /* Inference */
    HeatMapGenerator hmap_generator = HeatMapGenerator(context, precision);
    hmap_generator.Init(model_path);
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    cv::Mat heatmap = hmap_generator.Infer(image);
    cv::imwrite("./heatmap.png", heatmap);
    std::vector<Hotspot> hotspots = hmap_generator.LocateHotspots(heatmap);

    Hotspot max_hotspot = hmap_generator.LocateMaxHotspot(heatmap);
    printf("max_hot intensity: %f\n", max_hotspot.mean_intensity);

    /* Blend image and heatmap */
    cv::Mat blend_img;
    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    cv::cvtColor(heatmap, heatmap, cv::COLOR_RGB2BGR);
    cv::addWeighted(image, 0.5, heatmap, 0.5, 0, blend_img);
    cv::imwrite("./blend.png", blend_img);

    return 0;
}
