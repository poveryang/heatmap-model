#include "yolo_detector.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& begin, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void configure_threshold_from_env(const char* name, float& threshold) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') {
        return;
    }
    errno = 0;
    char* end = nullptr;
    const float value = std::strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || value < 0.f || value > 1.f) {
        fprintf(stderr, "Ignoring invalid %s=%s; expected a value in [0, 1]\n", name, text);
        return;
    }
    threshold = value;
}

int clamp_int(int value, int low, int high) {
    return std::max(low, std::min(value, high));
}

void dump_tensor_summary(const char* direction, int index, tensor_t tensor) {
    if (tensor == nullptr) {
        return;
    }
    int dims[8] = {};
    const int rank = get_tensor_shape(tensor, dims, 8);
    fprintf(stderr, " %s%d=%s[", direction, index, get_tensor_name(tensor));
    for (int i = 0; i < rank; ++i) {
        fprintf(stderr, "%s%d", i == 0 ? "" : "x", dims[i]);
    }
    fprintf(stderr, "]");
}

void dump_graph_summary(graph_t graph) {
    const int node_count = get_graph_node_num(graph);
    fprintf(stderr, "graph_summary nodes=%d\n", node_count);
    int timvx_nodes = 0;
    int cpu_nodes = 0;
    int timvx_subgraphs = 0;
    int cpu_subgraphs = 0;
    std::string previous_device;
    std::map<std::string, int> cpu_ops;
    for (int i = 0; i < node_count; ++i) {
        node_t node = get_graph_node_by_idx(graph, i);
        if (node == nullptr) {
            continue;
        }
        const char* device_name = get_node_device(node);
        const std::string device = device_name == nullptr ? "unassigned" : device_name;
        const char* op = get_node_op(node);
        if (device == "TIMVX") {
            ++timvx_nodes;
            if (previous_device != device) {
                ++timvx_subgraphs;
            }
        } else if (device == "CPU") {
            ++cpu_nodes;
            ++cpu_ops[op];
            if (previous_device != device) {
                ++cpu_subgraphs;
            }
        }
        previous_device = device;
        fprintf(stderr, "graph_node index=%d name=%s op=%s device=%s", i,
                get_node_name(node), op, device.c_str());
        const int input_count = get_node_input_number(node);
        const int output_count = get_node_output_number(node);
        for (int j = 0; j < input_count; ++j) {
            tensor_t tensor = get_node_input_tensor(node, j);
            dump_tensor_summary("in", j, tensor);
            release_graph_tensor(tensor);
        }
        for (int j = 0; j < output_count; ++j) {
            tensor_t tensor = get_node_output_tensor(node, j);
            dump_tensor_summary("out", j, tensor);
            release_graph_tensor(tensor);
        }
        fprintf(stderr, "\n");
        release_graph_node(node);
    }
    fprintf(stderr,
            "graph_partition_summary timvx_subgraphs=%d cpu_subgraphs=%d "
            "timvx_nodes=%d cpu_nodes=%d cpu_ops=",
            timvx_subgraphs, cpu_subgraphs, timvx_nodes, cpu_nodes);
    bool first = true;
    for (const auto& entry : cpu_ops) {
        fprintf(stderr, "%s%s:%d", first ? "" : ",", entry.first.c_str(), entry.second);
        first = false;
    }
    fprintf(stderr, "%s\n", first ? "none" : "");
    fflush(stderr);
}

}  // namespace

YoloDetector::YoloDetector(const std::string& context_name, const std::string& precision)
    : context_name_(context_name), precision_(precision) {}

YoloDetector::~YoloDetector() {
    if (graph_ != nullptr) {
        if (graph_prerun_) {
            postrun_graph(graph_);
        }
        destroy_graph(graph_);
    }
    if (context_ != nullptr) {
        destroy_context(context_);
    }
    if (tengine_initialized_) {
        release_tengine();
    }
}

bool YoloDetector::Init(const std::string& model_path) {
    configure_threshold_from_env("YOLO_CONF_THRESHOLD", confidence_threshold_);
    configure_threshold_from_env("YOLO_NMS_THRESHOLD", nms_iou_threshold_);
    int precision_mode = TENGINE_MODE_UINT8;
    int input_unit_size = 1;
    if (precision_ == "fp32") {
        precision_mode = TENGINE_MODE_FP32;
        input_unit_size = static_cast<int>(sizeof(float));
    } else if (precision_ != "uint8") {
        fprintf(stderr, "Unsupported precision: %s\n", precision_.c_str());
        return false;
    }

    if (init_tengine() != 0) {
        fprintf(stderr, "init_tengine failed\n");
        return false;
    }
    tengine_initialized_ = true;

    options_.num_thread = 4;
    options_.cluster = TENGINE_CLUSTER_ALL;
    options_.precision = precision_mode;
    options_.affinity = 0;

    if (context_name_ == "timvx") {
        context_ = create_context("timvx", 1);
        if (context_ == nullptr || set_context_device(context_, "TIMVX", nullptr, 0) < 0) {
            fprintf(stderr, "Create TIM-VX context failed\n");
            return false;
        }
    } else if (context_name_ != "cpu") {
        fprintf(stderr, "Unsupported context: %s\n", context_name_.c_str());
        return false;
    }

    graph_ = create_graph(context_, "tengine", model_path.c_str());
    if (graph_ == nullptr) {
        fprintf(stderr, "Create graph failed: %s\n", model_path.c_str());
        return false;
    }

    input_tensor_ = get_graph_input_tensor(graph_, 0, 0);
    if (input_tensor_ == nullptr) {
        fprintf(stderr, "Get input tensor failed\n");
        return false;
    }

    int model_input_dims[4] = {};
    const int model_input_rank = get_tensor_shape(input_tensor_, model_input_dims, 4);
    if (model_input_rank != 4 || model_input_dims[0] != 1 ||
        (model_input_dims[1] != 1 && model_input_dims[1] != 3)) {
        fprintf(stderr, "Unsupported model input shape\n");
        return false;
    }
    input_channels_ = model_input_dims[1];
    input_height_ = model_input_dims[2];
    input_width_ = model_input_dims[3];
    const int input_dims[4] = {1, input_channels_, input_height_, input_width_};
    if (set_tensor_shape(input_tensor_, input_dims, 4) < 0) {
        fprintf(stderr, "Set input tensor shape failed\n");
        return false;
    }

    const int input_elements = input_channels_ * input_height_ * input_width_;
    void* input_buffer = nullptr;
    if (precision_ == "uint8") {
        input_uint8_.resize(input_elements);
        input_buffer = input_uint8_.data();
    } else {
        input_float_.resize(input_elements);
        input_buffer = input_float_.data();
    }
    if (set_tensor_buffer(input_tensor_, input_buffer, input_elements * input_unit_size) < 0) {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return false;
    }

    if (prerun_graph_multithread(graph_, options_) < 0) {
        fprintf(stderr, "Prerun graph failed\n");
        return false;
    }
    graph_prerun_ = true;

    if (std::getenv("YOLO_DUMP_GRAPH") != nullptr) {
        dump_graph_summary(graph_);
    }

    get_tensor_quant_param(input_tensor_, &input_scale_, &input_zero_point_, 1);
    const int output_node_count = get_graph_output_node_number(graph_);
    if (output_node_count > 2) {
        std::map<int, RawHeadLevel> levels;
        for (int index = 0; index < output_node_count; ++index) {
            tensor_t tensor = get_graph_output_tensor(graph_, index, 0);
            int dims[4] = {};
            const int rank = get_tensor_shape(tensor, dims, 4);
            if ((rank != 3 && rank != 4) || dims[0] != 1) {
                fprintf(stderr, "Unsupported per-level output shape at index %d\n", index);
                return false;
            }
            const int side = rank == 3 ? static_cast<int>(std::sqrt(dims[2])) : dims[2];
            if (side <= 0 || (rank == 3 && side * side != dims[2]) ||
                (rank == 4 && (dims[2] != dims[3] || dims[2] <= 0))) {
                fprintf(stderr, "Unsupported per-level spatial shape at index %d\n", index);
                return false;
            }
            RawHeadTensor output;
            output.tensor = tensor;
            output.channels = dims[1];
            output.height = side;
            output.width = side;
            output.data_type = get_tensor_data_type(tensor);
            get_tensor_quant_param(tensor, &output.scale, &output.zero_point, 1);
            const int stride = input_width_ / output.width;
            RawHeadLevel& level = levels[stride];
            level.stride = stride;
            if (output.channels == 3) {
                level.scores = output;
            } else if (output.channels >= 4 && output.channels % 4 == 0) {
                level.boxes = output;
            } else {
                fprintf(stderr, "Unsupported per-level output channels: %d\n", output.channels);
                return false;
            }
        }
        for (const auto& entry : levels) {
            if (entry.second.boxes.tensor == nullptr || entry.second.scores.tensor == nullptr) {
                fprintf(stderr, "Incomplete raw head at stride %d\n", entry.first);
                return false;
            }
            raw_head_levels_.push_back(entry.second);
        }
    }

    if (raw_head_levels_.empty()) {
    boxes_tensor_ = get_graph_output_tensor(graph_, 0, 0);
    scores_tensor_ = get_graph_output_tensor(graph_, 1, 0);
    if (boxes_tensor_ == nullptr || scores_tensor_ == nullptr) {
        fprintf(stderr, "Get YOLO boxes/scores output tensors failed\n");
        return false;
    }
    boxes_dim_count_ = get_tensor_shape(boxes_tensor_, boxes_dims_, 4);
    scores_dim_count_ = get_tensor_shape(scores_tensor_, scores_dims_, 4);
    if (boxes_dim_count_ < 3 || scores_dim_count_ < 3) {
        fprintf(stderr, "Unsupported YOLO output ranks: boxes=%d scores=%d\n",
                boxes_dim_count_, scores_dim_count_);
        return false;
    }
    boxes_data_type_ = get_tensor_data_type(boxes_tensor_);
    scores_data_type_ = get_tensor_data_type(scores_tensor_);
    get_tensor_quant_param(boxes_tensor_, &boxes_scale_, &boxes_zero_point_, 1);
    get_tensor_quant_param(scores_tensor_, &scores_scale_, &scores_zero_point_, 1);
    boxes_elements_ = 1;
    for (int i = 0; i < boxes_dim_count_; ++i) {
        boxes_elements_ *= boxes_dims_[i];
    }
    scores_elements_ = 1;
    for (int i = 0; i < scores_dim_count_; ++i) {
        scores_elements_ *= scores_dims_[i];
    }
    }

    fprintf(stdout, "model_info context=%s precision=%s input=1x%dx%dx%d "
            "raw_head_levels=%zu "
            "boxes_elements=%d scores_elements=%d input_q=(%.9g,%d) "
            "boxes_q=(%.9g,%d) scores_q=(%.9g,%d)\n",
            context_name_.c_str(), precision_.c_str(), input_channels_, input_height_, input_width_,
            raw_head_levels_.size(),
            boxes_elements_, scores_elements_,
            input_scale_, input_zero_point_, boxes_scale_, boxes_zero_point_,
            scores_scale_, scores_zero_point_);
    return true;
}

double YoloDetector::Warmup(int count) {
    if (!graph_prerun_ || count <= 0) {
        return 0.;
    }
    if (precision_ == "uint8") {
        std::fill(input_uint8_.begin(), input_uint8_.end(),
                  static_cast<uint8_t>(clamp_int(input_zero_point_, 0, 255)));
    } else {
        std::fill(input_float_.begin(), input_float_.end(), 0.f);
    }

    const auto begin = Clock::now();
    for (int i = 0; i < count; ++i) {
        if (run_graph(graph_, 1) < 0) {
            fprintf(stderr, "Warmup inference failed at iteration %d\n", i);
            return -1.;
        }
    }
    return elapsed_ms(begin, Clock::now()) / count;
}

bool YoloDetector::Infer(const cv::Mat& image,
                         std::vector<YoloDetection>& detections,
                         YoloTiming& timing) {
    if (!graph_prerun_ || image.empty()) {
        return false;
    }

    const auto total_begin = Clock::now();
    LetterboxInfo info;
    const auto preprocess_begin = Clock::now();
    if (!Preprocess(image, info)) {
        return false;
    }
    const auto preprocess_end = Clock::now();

    const auto inference_begin = Clock::now();
    if (run_graph(graph_, 1) < 0) {
        fprintf(stderr, "Run graph failed\n");
        return false;
    }
    const auto inference_end = Clock::now();

    const auto postprocess_begin = Clock::now();
    if (!Decode(info, detections)) {
        return false;
    }
    const auto postprocess_end = Clock::now();

    timing.preprocess_ms = elapsed_ms(preprocess_begin, preprocess_end);
    timing.inference_ms = elapsed_ms(inference_begin, inference_end);
    timing.postprocess_ms = elapsed_ms(postprocess_begin, postprocess_end);
    timing.total_ms = elapsed_ms(total_begin, postprocess_end);
    return true;
}

bool YoloDetector::Preprocess(const cv::Mat& image, LetterboxInfo& info) {
    cv::Mat input_image;
    if (input_channels_ == 1 && image.channels() == 1) {
        input_image = image;
    } else if (input_channels_ == 1 && image.channels() == 3) {
        cv::cvtColor(image, input_image, cv::COLOR_BGR2GRAY);
    } else if (input_channels_ == 1 && image.channels() == 4) {
        cv::cvtColor(image, input_image, cv::COLOR_BGRA2GRAY);
    } else if (input_channels_ == 3 && image.channels() == 1) {
        cv::cvtColor(image, input_image, cv::COLOR_GRAY2RGB);
    } else if (input_channels_ == 3 && image.channels() == 3) {
        cv::cvtColor(image, input_image, cv::COLOR_BGR2RGB);
    } else if (input_channels_ == 3 && image.channels() == 4) {
        cv::cvtColor(image, input_image, cv::COLOR_BGRA2RGB);
    } else {
        fprintf(stderr, "Unsupported input channels: %d\n", image.channels());
        return false;
    }

    info.original_width = input_image.cols;
    info.original_height = input_image.rows;
    info.scale = std::min(input_width_ / static_cast<float>(input_image.cols),
                          input_height_ / static_cast<float>(input_image.rows));
    const int resized_width = std::max(1, static_cast<int>(std::round(input_image.cols * info.scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(input_image.rows * info.scale)));
    info.left = (input_width_ - resized_width) / 2;
    info.top = (input_height_ - resized_height) / 2;

    cv::Mat resized;
    cv::resize(input_image, resized, cv::Size(resized_width, resized_height), 0., 0.,
               cv::INTER_LINEAR);
    const int canvas_type = input_channels_ == 1 ? CV_8UC1 : CV_8UC3;
    cv::Mat canvas(input_height_, input_width_, canvas_type, cv::Scalar::all(114));
    resized.copyTo(canvas(cv::Rect(info.left, info.top, resized_width, resized_height)));

    std::vector<cv::Mat> channels;
    if (input_channels_ == 1) {
        channels.push_back(canvas);
    } else {
        cv::split(canvas, channels);
    }
    const size_t plane = static_cast<size_t>(input_width_) * input_height_;
    for (int c = 0; c < input_channels_; ++c) {
        cv::Mat converted;
        if (precision_ == "uint8") {
            channels[c].convertTo(converted, CV_8U, 1.f / (255.f * input_scale_),
                                  input_zero_point_);
            std::memcpy(input_uint8_.data() + c * plane, converted.data, plane);
        } else {
            channels[c].convertTo(converted, CV_32F, 1.f / 255.f);
            std::memcpy(input_float_.data() + c * plane, converted.ptr<float>(),
                        plane * sizeof(float));
        }
    }
    return true;
}

float YoloDetector::OutputValue(const void* output, int index, int data_type,
                                float scale, int zero_point) {
    if (data_type == TENGINE_DT_FP32) {
        return static_cast<const float*>(output)[index];
    }
    if (data_type == TENGINE_DT_UINT8) {
        const auto value = static_cast<const uint8_t*>(output)[index];
        return (static_cast<int>(value) - zero_point) * scale;
    }
    if (data_type == TENGINE_DT_INT8) {
        const auto value = static_cast<const int8_t*>(output)[index];
        return (static_cast<int>(value) - zero_point) * scale;
    }
    return 0.f;
}

bool YoloDetector::Decode(const LetterboxInfo& info, std::vector<YoloDetection>& detections) const {
    if (!raw_head_levels_.empty()) {
        return DecodePerLevel(info, detections);
    }
    const void* boxes = get_tensor_buffer(boxes_tensor_);
    const void* scores = get_tensor_buffer(scores_tensor_);
    if (boxes == nullptr || scores == nullptr) {
        fprintf(stderr, "Get YOLO boxes/scores buffers failed\n");
        return false;
    }

    const char* dump_directory = std::getenv("YOLO_DUMP_OUTPUTS");
    if (dump_directory != nullptr) {
        auto dump_output = [&](const char* name, const void* data, int elements, int data_type,
                               float scale, int zero_point) {
            std::vector<float> values(elements);
            float minimum = 0.f;
            float maximum = 0.f;
            double sum = 0.;
            for (int i = 0; i < elements; ++i) {
                values[i] = OutputValue(data, i, data_type, scale, zero_point);
                if (i == 0 || values[i] < minimum) minimum = values[i];
                if (i == 0 || values[i] > maximum) maximum = values[i];
                sum += values[i];
            }
            const std::string path = std::string(dump_directory) + "/" + name + ".f32";
            std::ofstream stream(path, std::ios::binary);
            stream.write(reinterpret_cast<const char*>(values.data()),
                         static_cast<std::streamsize>(values.size() * sizeof(float)));
            printf("output_dump name=%s elements=%d min=%.6f max=%.6f mean=%.6f path=%s\n",
                   name, elements, minimum, maximum, sum / elements, path.c_str());
        };
        dump_output("output0", boxes, boxes_elements_, boxes_data_type_,
                    boxes_scale_, boxes_zero_point_);
        dump_output("output1", scores, scores_elements_, scores_data_type_,
                    scores_scale_, scores_zero_point_);
    }

    const bool boxes_channels_first = boxes_dims_[1] >= 4 && boxes_dims_[1] % 4 == 0;
    const bool scores_channels_first = scores_dims_[1] == 3;
    const bool boxes_channels_last = boxes_dims_[boxes_dim_count_ - 1] >= 4 &&
                                     boxes_dims_[boxes_dim_count_ - 1] % 4 == 0;
    const bool scores_channels_last = scores_dims_[scores_dim_count_ - 1] == 3;
    if ((!boxes_channels_first && !boxes_channels_last) ||
        (!scores_channels_first && !scores_channels_last)) {
        fprintf(stderr, "Unsupported YOLO dual-output shapes\n");
        return false;
    }
    const int box_channels = boxes_channels_first ? boxes_dims_[1]
                                                  : boxes_dims_[boxes_dim_count_ - 1];
    const int reg_max = box_channels / 4;
    const int box_candidates = boxes_elements_ / boxes_dims_[0] / box_channels;
    const int score_candidates = scores_elements_ / scores_dims_[0] / 3;
    if ((box_candidates != 8400 && box_candidates != 34000) ||
        score_candidates != box_candidates || (reg_max != 1 && reg_max != 16)) {
        fprintf(stderr, "Unsupported YOLO candidate counts: boxes=%d scores=%d\n",
                box_candidates, score_candidates);
        return false;
    }
    const int candidates = box_candidates;

    auto box_at = [&](int candidate, int channel) {
        const int index = boxes_channels_first ? channel * candidates + candidate
                                               : candidate * box_channels + channel;
        return OutputValue(boxes, index, boxes_data_type_, boxes_scale_, boxes_zero_point_);
    };
    auto score_at = [&](int candidate, int channel) {
        const int index = scores_channels_first ? channel * candidates + candidate
                                                : candidate * 3 + channel;
        return OutputValue(scores, index, scores_data_type_, scores_scale_, scores_zero_point_);
    };
    auto sigmoid = [](float value) {
        if (value >= 0.f) {
            return 1.f / (1.f + std::exp(-value));
        }
        const float exponential = std::exp(value);
        return exponential / (1.f + exponential);
    };
    auto distance_at = [&](int candidate, int side) {
        if (reg_max == 1) {
            return box_at(candidate, side);
        }
        float maximum = box_at(candidate, side * reg_max);
        for (int bin = 1; bin < reg_max; ++bin) {
            maximum = std::max(maximum, box_at(candidate, side * reg_max + bin));
        }
        float denominator = 0.f;
        float expected = 0.f;
        for (int bin = 0; bin < reg_max; ++bin) {
            const float probability = std::exp(box_at(candidate, side * reg_max + bin) - maximum);
            denominator += probability;
            expected += probability * static_cast<float>(bin);
        }
        return expected / denominator;
    };

    const std::vector<int> strides = candidates == 34000
                                         ? std::vector<int>{4, 8, 16, 32}
                                         : std::vector<int>{8, 16, 32};

    detections.clear();
    detections.reserve(128);
    float maximum_score = 0.f;
    int maximum_candidate = -1;
    int maximum_class = -1;
    for (int i = 0; i < candidates; ++i) {
        int class_id = 0;
        float score = sigmoid(score_at(i, 0));
        for (int c = 1; c < 3; ++c) {
            const float class_score = sigmoid(score_at(i, c));
            if (class_score > score) {
                score = class_score;
                class_id = c;
            }
        }
        if (score > maximum_score) {
            maximum_score = score;
            maximum_candidate = i;
            maximum_class = class_id;
        }
        if (score < confidence_threshold_) {
            continue;
        }

        int level_index = i;
        int grid_width = 0;
        int grid_height = 0;
        float stride = 0.f;
        for (const int candidate_stride : strides) {
            grid_width = input_width_ / candidate_stride;
            grid_height = input_height_ / candidate_stride;
            const int level_candidates = grid_width * grid_height;
            if (level_index < level_candidates) {
                stride = static_cast<float>(candidate_stride);
                break;
            }
            level_index -= level_candidates;
        }
        if (stride == 0.f || level_index >= grid_width * grid_height) {
            fprintf(stderr, "Failed to resolve YOLO feature level for candidate %d\n", i);
            return false;
        }
        const float anchor_x = static_cast<float>(level_index % grid_width) + 0.5f;
        const float anchor_y = static_cast<float>(level_index / grid_width) + 0.5f;
        const float left = distance_at(i, 0);
        const float top = distance_at(i, 1);
        const float right = distance_at(i, 2);
        const float bottom = distance_at(i, 3);
        const float center_x = (anchor_x + (right - left) * 0.5f) * stride;
        const float center_y = (anchor_y + (bottom - top) * 0.5f) * stride;
        const float width = (left + right) * stride;
        const float height = (top + bottom) * stride;
        float x0 = (center_x - width * 0.5f - info.left) / info.scale;
        float y0 = (center_y - height * 0.5f - info.top) / info.scale;
        float x1 = (center_x + width * 0.5f - info.left) / info.scale;
        float y1 = (center_y + height * 0.5f - info.top) / info.scale;
        x0 = std::max(0.f, std::min(x0, static_cast<float>(info.original_width - 1)));
        y0 = std::max(0.f, std::min(y0, static_cast<float>(info.original_height - 1)));
        x1 = std::max(0.f, std::min(x1, static_cast<float>(info.original_width - 1)));
        y1 = std::max(0.f, std::min(y1, static_cast<float>(info.original_height - 1)));
        if (x1 <= x0 || y1 <= y0) {
            continue;
        }

        YoloDetection detection;
        detection.box = cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
        detection.score = score;
        detection.class_id = class_id;
        detections.push_back(detection);
    }
    if (std::getenv("YOLO_DEBUG") != nullptr) {
        printf("decode_stats max_score=%.6f candidate=%d class=%d pre_nms=%zu\n",
               maximum_score, maximum_candidate, maximum_class, detections.size());
    }
    Nms(detections, nms_iou_threshold_);
    return true;
}

bool YoloDetector::DecodePerLevel(const LetterboxInfo& info,
                                  std::vector<YoloDetection>& detections) const {
    auto sigmoid = [](float value) {
        if (value >= 0.f) return 1.f / (1.f + std::exp(-value));
        const float exponential = std::exp(value);
        return exponential / (1.f + exponential);
    };

    detections.clear();
    detections.reserve(128);
    float maximum_score = 0.f;
    int maximum_candidate = -1;
    int maximum_class = -1;
    int candidate_offset = 0;
    for (const RawHeadLevel& level : raw_head_levels_) {
        const void* boxes = get_tensor_buffer(level.boxes.tensor);
        const void* scores = get_tensor_buffer(level.scores.tensor);
        if (boxes == nullptr || scores == nullptr) {
            fprintf(stderr, "Get per-level YOLO output buffers failed at stride %d\n", level.stride);
            return false;
        }
        const int candidates = level.boxes.height * level.boxes.width;
        const int reg_max = level.boxes.channels / 4;
        auto box_at = [&](int candidate, int channel) {
            return OutputValue(boxes, channel * candidates + candidate,
                               level.boxes.data_type, level.boxes.scale,
                               level.boxes.zero_point);
        };
        auto score_at = [&](int candidate, int channel) {
            return OutputValue(scores, channel * candidates + candidate,
                               level.scores.data_type, level.scores.scale,
                               level.scores.zero_point);
        };
        auto distance_at = [&](int candidate, int side) {
            if (reg_max == 1) return box_at(candidate, side);
            float maximum = box_at(candidate, side * reg_max);
            for (int bin = 1; bin < reg_max; ++bin) {
                maximum = std::max(maximum, box_at(candidate, side * reg_max + bin));
            }
            float denominator = 0.f;
            float expected = 0.f;
            for (int bin = 0; bin < reg_max; ++bin) {
                const float probability = std::exp(box_at(candidate, side * reg_max + bin) - maximum);
                denominator += probability;
                expected += probability * static_cast<float>(bin);
            }
            return expected / denominator;
        };

        for (int candidate = 0; candidate < candidates; ++candidate) {
            int class_id = 0;
            float score = sigmoid(score_at(candidate, 0));
            for (int class_index = 1; class_index < 3; ++class_index) {
                const float candidate_score = sigmoid(score_at(candidate, class_index));
                if (candidate_score > score) {
                    score = candidate_score;
                    class_id = class_index;
                }
            }
            if (score > maximum_score) {
                maximum_score = score;
                maximum_candidate = candidate_offset + candidate;
                maximum_class = class_id;
            }
            if (score < confidence_threshold_) continue;

            const float anchor_x = static_cast<float>(candidate % level.boxes.width) + 0.5f;
            const float anchor_y = static_cast<float>(candidate / level.boxes.width) + 0.5f;
            const float stride = static_cast<float>(level.stride);
            const float x0_input = (anchor_x - distance_at(candidate, 0)) * stride;
            const float y0_input = (anchor_y - distance_at(candidate, 1)) * stride;
            const float x1_input = (anchor_x + distance_at(candidate, 2)) * stride;
            const float y1_input = (anchor_y + distance_at(candidate, 3)) * stride;
            float x0 = (x0_input - info.left) / info.scale;
            float y0 = (y0_input - info.top) / info.scale;
            float x1 = (x1_input - info.left) / info.scale;
            float y1 = (y1_input - info.top) / info.scale;
            x0 = std::max(0.f, std::min(x0, static_cast<float>(info.original_width - 1)));
            y0 = std::max(0.f, std::min(y0, static_cast<float>(info.original_height - 1)));
            x1 = std::max(0.f, std::min(x1, static_cast<float>(info.original_width - 1)));
            y1 = std::max(0.f, std::min(y1, static_cast<float>(info.original_height - 1)));
            if (x1 <= x0 || y1 <= y0) continue;
            detections.push_back({cv::Rect2f(x0, y0, x1 - x0, y1 - y0), score, class_id});
        }
        candidate_offset += candidates;
    }
    if (std::getenv("YOLO_DEBUG") != nullptr) {
        printf("decode_stats max_score=%.6f candidate=%d class=%d pre_nms=%zu\n",
               maximum_score, maximum_candidate, maximum_class, detections.size());
    }
    Nms(detections, nms_iou_threshold_);
    return true;
}

float YoloDetector::IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float intersection = (a & b).area();
    const float union_area = a.area() + b.area() - intersection;
    return union_area > 0.f ? intersection / union_area : 0.f;
}

void YoloDetector::Nms(std::vector<YoloDetection>& detections, float iou_threshold) {
    std::sort(detections.begin(), detections.end(),
              [](const YoloDetection& a, const YoloDetection& b) { return a.score > b.score; });
    std::vector<YoloDetection> kept;
    kept.reserve(detections.size());
    for (const auto& candidate : detections) {
        bool suppressed = false;
        for (const auto& selected : kept) {
            if (candidate.class_id == selected.class_id && IoU(candidate.box, selected.box) > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
        }
        if (kept.size() >= 300) {
            break;
        }
    }
    detections.swap(kept);
}

const char* YoloDetector::ClassName(int class_id) {
    static const char* names[] = {"bar", "qr", "dm"};
    return class_id >= 0 && class_id < 3 ? names[class_id] : "unknown";
}

void YoloDetector::DrawDetections(cv::Mat& image, const std::vector<YoloDetection>& detections) {
    static const cv::Scalar colors[] = {
        cv::Scalar(36, 176, 72),
        cv::Scalar(224, 144, 32),
        cv::Scalar(190, 58, 210),
    };
    for (const auto& detection : detections) {
        const cv::Scalar color = colors[std::max(0, std::min(detection.class_id, 2))];
        cv::rectangle(image, detection.box, color, 2, cv::LINE_AA);
        char label[96];
        snprintf(label, sizeof(label), "%s %.2f", ClassName(detection.class_id), detection.score);
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        const int x = std::max(0, static_cast<int>(detection.box.x));
        const int y = std::max(text_size.height + baseline + 2, static_cast<int>(detection.box.y));
        const cv::Rect background(x, y - text_size.height - baseline - 2,
                                  text_size.width + 6, text_size.height + baseline + 2);
        cv::rectangle(image, background, color, cv::FILLED);
        cv::putText(image, label, cv::Point(x + 3, y - baseline - 1),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}
