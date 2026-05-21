#include "hmap_generator.h"

#include <cmath>


HeatMapGenerator::HeatMapGenerator(const std::string &context_name, const std::string &precision) {
    context_name_ = context_name;
    precision_ = precision;
}

void HeatMapGenerator::Init(const std::string &model_path) {
    /* Set precision */
    int unit_size;
    int opt_precision;
    if (precision_ == "fp32") {
        opt_precision = TENGINE_MODE_FP32;
        unit_size = 4;
    } else if (precision_ == "uint8") {
        opt_precision = TENGINE_MODE_UINT8;
        unit_size = 1;
    } else {
        fprintf(stderr, "Precision not supported.\n");
        exit(1);
    }

    /* init tengine */
    if (init_tengine() != 0) {
        fprintf(stderr, "Init tengine failed.\n");
        exit(1);
    }

    /*set runtime options*/
    opt_.num_thread = 1;
    opt_.cluster = TENGINE_CLUSTER_ALL;
    opt_.precision = opt_precision;
    opt_.affinity = 0;

    /* create context */
    context_t context = nullptr;
    if (context_name_ == "timvx") {
        /* create VeriSilicon TIM-VX backend */
        context = create_context("timvx", 1);
        int rtt = set_context_device(context, "TIMVX", nullptr, 0);
        if (rtt < 0) {
            fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
            exit(1);
        }
    }

    /* load model */
    graph_ = create_graph(context, "tengine", model_path.c_str());
    if (graph_ == nullptr) {
        fprintf(stderr, "Create graph failed.\n");
        exit(1);
    } else {
        fprintf(stdout, "Load HMap-Generator model & create graph success.\n");
    }

    /* get input tensor quantization params */
    input_tensor_ = get_graph_input_tensor(graph_, 0, 0);
    if (input_tensor_ == nullptr) {
        fprintf(stderr, "Get input tensor failed\n");
        exit(1);
    }
    get_tensor_quant_param(input_tensor_, &input_scale, &input_zero_point, 1);

    /* set input tensor shape */
    int img_h = 400;  // TODO: get from model
    int img_w = 640;
    int img_c = 1;
    input_buffer_size_ = img_h * img_w * img_c * unit_size;
    int in_dims[4] = {1, img_c, img_h, img_w}; // nchw
    if (set_tensor_shape(input_tensor_, in_dims, 4) < 0) {
        fprintf(stderr, "Set input tensor shape failed\n");
        exit(1);
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph_, opt_) < 0) {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        exit(1);
    }

    /* get output tensor quantization params */
    output_tensor_ = get_graph_output_tensor(graph_, 0, 0);
    if (get_tensor_shape(output_tensor_, out_dim_, 4) < 0) {
        fprintf(stderr, "Get output tensor shape failed\n");
        exit(1);
    }
    get_tensor_quant_param(output_tensor_, &output_scale, &output_zero_point, 1);
}

cv::Mat HeatMapGenerator::Infer(const cv::Mat &image) {
    /* preprocess */
    cv::Mat in_img = PreProcess(image);

    /* infer */
    cv::Mat heatmap;
    if (precision_ == "fp32") {
        heatmap = InferFP32(in_img);
    } else if (precision_ == "uint8") {
        heatmap = InferUInt8(in_img);
    } else {
        fprintf(stderr, "Precision not supported.\n");
        exit(1);
    }

    /* postprocess */
    cv::Size out_size = image.size();
    heatmap = PostProcess(heatmap, out_size);
    return heatmap;
}

std::vector<Hotspot> HeatMapGenerator::LocateHotspots(const cv::Mat &heatmap, double intensity_thres) {
    /* Split the heatmap into channels and iterate over each channel */
    std::vector<Hotspot> hotspots;
    std::vector<cv::Mat> heatmap_chs;
    cv::split(heatmap, heatmap_chs);
    for (int ch = 0; ch < heatmap_chs.size(); ch++) {
        cv::Mat heatmap_ch = heatmap_chs[ch];

        /* Filter out the channel with low max intensity */
        double max_ch_intensity;
        cv::minMaxLoc(heatmap_ch, nullptr, &max_ch_intensity);
        if (max_ch_intensity < intensity_thres) {
            continue;
        }

        /* Threshold the heatmap channel to get the binary hmap, and filter out the channel with low threshold */
        cv::Mat hmap_bin;
        double thresh = cv::threshold(heatmap_ch, hmap_bin, 0, 1, cv::THRESH_BINARY | cv::THRESH_OTSU);
        if ((thresh < intensity_thres / 2) && (thresh > 1e-9)) {
            continue;
        }

        /* Find the connected components in the binary hmap */
        cv::Mat labels, stats, centroids;
        cv::connectedComponentsWithStats(hmap_bin, labels, stats, centroids);

        /* Get ROI from the connected components and calculate the mean intensity of each ROI */
        for (int i = 1; i < stats.rows; i++) {
            // filter out the connected component with small area
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < 100) {  // TODO: make thresholds configurable
                continue;
            }

            cv::Point2d centroid = cv::Point2d(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
            cv::Rect2d rect(stats.at<int>(i, cv::CC_STAT_LEFT),
                            stats.at<int>(i, cv::CC_STAT_TOP),
                            stats.at<int>(i, cv::CC_STAT_WIDTH),
                            stats.at<int>(i, cv::CC_STAT_HEIGHT));
            double sum_intensity = cv::sum(heatmap_ch(rect))[0];
            double mean_intensity = sum_intensity / area;

            // store the hmap info
            Hotspot hotspot;
            hotspot.type = ch;
            hotspot.area = area;
            hotspot.centroid = centroid;
            hotspot.rect = rect;
            hotspot.sum_intensity = sum_intensity;
            hotspot.mean_intensity = mean_intensity;
            hotspot.mask = hmap_bin(rect);
            IdentifyHotspot(hotspot);
            hotspots.emplace_back(hotspot);
        }
    }

    return hotspots;
}

Hotspot HeatMapGenerator::LocateMaxHotspot(const cv::Mat &heatmap, double intensity_thres) {
    // create a new hotspot without any info
    Hotspot max_hotspot;

    /* Locate all the hotspots */
    std::vector<Hotspot> hotspots = LocateHotspots(heatmap, intensity_thres);

    /* Pick the hotspot with the highest intensity */
    for (const auto &hots: hotspots) {
        if (hots.sum_intensity > max_hotspot.sum_intensity) {
            max_hotspot = hots;
        }
    }
    return max_hotspot;
}

void HeatMapGenerator::IdentifyHotspot(Hotspot &new_hotspot) {
    /* Match the new hotspot with the existing hotspots */
    for (auto &exist_hotspot: hots_recoder) {
        // Determine if the new hotspot is the same as the existing hotspot
        if (new_hotspot.type != exist_hotspot.type) {
            continue;
        }

        double iou = CalcHotsIOU(new_hotspot, exist_hotspot);
        if (iou > 0.2) {  // TODO: make this threshold configurable
            new_hotspot.id = exist_hotspot.id;
            exist_hotspot.sum_intensity = std::max(new_hotspot.sum_intensity, exist_hotspot.sum_intensity);
            return;
        }
    }

    /* If the hotspot is not matched with any existing hotspot, assign a new id */
    new_hotspot.id = static_cast<int>(hots_recoder.size());
    hots_recoder.emplace_back(new_hotspot);
}

double HeatMapGenerator::CalcHotsIOU(const Hotspot &hot1, const Hotspot &hot2) {
    // rects coordination
    cv::Rect2d rect1 = hot1.rect;
    cv::Rect2d rect2 = hot2.rect;
    double x11 = rect1.x;
    double y11 = rect1.y;
    double x12 = rect1.x + rect1.width;
    double y12 = rect1.y + rect1.height;
    double x21 = rect2.x;
    double y21 = rect2.y;
    double x22 = rect2.x + rect2.width;
    double y22 = rect2.y + rect2.height;

    // intersection rect
    cv::Rect2d int_rect1;
    cv::Rect2d int_rect2;
    double x = std::max(x11, x21);
    double y = std::max(y11, y21);
    double width = std::min(x12, x22) - x;
    double height = std::min(y12, y22) - y;
    cv::Rect2d int_rect = cv::Rect2d(x, y, width, height);
    int_rect1 = int_rect - cv::Point2d(x11, y11);
    int_rect2 = int_rect - cv::Point2d(x21, y21);
    if (int_rect1.width <= 0 || int_rect1.height <= 0 || int_rect2.width <= 0 || int_rect2.height <= 0) {
        return 0;
    }

    // intersection area
    cv::Mat mask1 = hot1.mask;
    cv::Mat mask2 = hot2.mask;
    cv::Mat mask_int = mask1(int_rect1) & mask2(int_rect2);

    double area1 = hot1.area;
    double area2 = hot2.area;
    double area_int = cv::sum(mask_int)[0];
    double iou = area_int / (area1 + area2 - area_int);
    return iou;
}

cv::Mat HeatMapGenerator::InferFP32(cv::Mat &image) {
    /* 1. set image data to input tensor */
    auto *image_data = image.data;
    if (set_tensor_buffer(input_tensor_, image_data, input_buffer_size_) < 0) {
        fprintf(stderr, "Set input tensor buffer failed\n");
        exit(1);
    }

    /* 2. run the graph */
    if (run_graph(graph_, 1) < 0) {
        fprintf(stderr, "Run graph failed.\n");
        exit(1);
    }

    /* 3. get the heatmap data from output tensor and transform to cv::Mat */
    auto output_fp32 = (float *) get_tensor_buffer(output_tensor_);
    if (output_fp32 == nullptr) {
        fprintf(stderr, "Get output data failed\n");
        exit(1);
    }
    cv::Mat heatmap = OutputTensorToMat(output_fp32, CV_32F);

    return heatmap;
}

cv::Mat HeatMapGenerator::InferUInt8(cv::Mat &image) {
    /* 1. set image data to input tensor */
    auto *image_data = image.data;
    if (set_tensor_buffer(input_tensor_, image_data, input_buffer_size_) < 0) {
        fprintf(stderr, "Set input tensor buffer failed\n");
        exit(1);
    }

    /* 2. run the graph */
    if (run_graph(graph_, 1) < 0) {
        fprintf(stderr, "Run graph failed.\n");
        exit(1);
    }

    /* 3. get the heatmap sample_images from output tensor and transform to cv::Mat */
    auto output_uint8 = (uint8_t *) get_tensor_buffer(output_tensor_);
    if (output_uint8 == nullptr) {
        fprintf(stderr, "Get output sample_images failed\n");
        exit(1);
    }
    cv::Mat heatmap = OutputTensorToMat(output_uint8, CV_8U);

    return heatmap;
}

cv::Mat HeatMapGenerator::PreProcess(const cv::Mat &image) {
    cv::Mat dst_img;
    cv::resize(image, dst_img, cv::Size(640, 400));  // TODO:
    if (precision_ == "fp32") {
        dst_img.convertTo(dst_img, CV_32FC3, 1.0 / 255);
        dst_img = (dst_img - cv::Scalar(0.4329)) / cv::Scalar(0.2349);
    }
//    else if (precision_ == "uint8"){
//        dst_img = dst_img * input_scale + input_zero_point;
//    }
    return dst_img;
}

cv::Mat HeatMapGenerator::PostProcess(const cv::Mat &image, cv::Size &dst_size) {
    cv::Mat dst_img;
    cv::resize(image, dst_img, dst_size, 0, 0, cv::INTER_NEAREST);

    if (precision_ == "fp32") {
        dst_img.convertTo(dst_img, CV_32FC3);
        dst_img = Sigmoid(dst_img); // sigmoid
        dst_img.convertTo(dst_img, CV_8UC3, 255);
    } else if (precision_ == "uint8") {
        dst_img.convertTo(dst_img, CV_32FC3);
        dst_img = (dst_img - cv::Scalar(output_zero_point, output_zero_point, output_zero_point))
                * output_scale;
//        dst_img = (dst_img - output_zero_point) * output_scale;
        dst_img = Sigmoid(dst_img); // sigmoid
        dst_img.convertTo(dst_img, CV_8UC3, 255);
    }
    return dst_img;
}

cv::Mat HeatMapGenerator::Sigmoid(const cv::Mat &image) {
    cv::Mat dst_img = cv::Mat(image.rows, image.cols, CV_32FC3);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < image.channels(); k++) {
                float value = image.at<cv::Vec3f>(i, j)[k];
                if (value >= 0) {
                    dst_img.at<cv::Vec3f>(i, j)[k] = 1.f / (1.f + std::exp(-value));
                } else {
                    float exp_value = std::exp(value);
                    dst_img.at<cv::Vec3f>(i, j)[k] = exp_value / (1.f + exp_value);
                }
            }
        }
    }
    return dst_img;
}

cv::Mat HeatMapGenerator::OutputTensorToMat(const void *data, int depth) const {
    if (out_dim_[0] != 1) {
        fprintf(stderr, "Unsupported output tensor batch size: %d\n", out_dim_[0]);
        exit(1);
    }

    const int channels = 3;
    const int single_channel_type = CV_MAKETYPE(depth, 1);
    const int three_channel_type = CV_MAKETYPE(depth, channels);

    if (out_dim_[1] == channels) {
        int height = out_dim_[2];
        int width = out_dim_[3];
        size_t plane_bytes = static_cast<size_t>(height) * width * CV_ELEM_SIZE1(depth);
        auto *base = const_cast<unsigned char *>(static_cast<const unsigned char *>(data));

        std::vector<cv::Mat> planes;
        planes.reserve(channels);
        for (int ch = 0; ch < channels; ch++) {
            planes.emplace_back(height, width, single_channel_type, base + plane_bytes * ch);
        }

        cv::Mat heatmap;
        cv::merge(planes, heatmap);
        return heatmap;
    }

    if (out_dim_[3] == channels) {
        return cv::Mat(out_dim_[1], out_dim_[2], three_channel_type, const_cast<void *>(data)).clone();
    }

    fprintf(stderr, "Unsupported output tensor shape: [%d, %d, %d, %d]\n",
            out_dim_[0], out_dim_[1], out_dim_[2], out_dim_[3]);
    exit(1);
}

HeatMapGenerator::~HeatMapGenerator() {
    /* release tengine */
    postrun_graph(graph_);
    destroy_graph(graph_);
    release_tengine();
}
