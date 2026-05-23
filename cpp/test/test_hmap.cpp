#include <algorithm>
#include <cctype>
#include <dirent.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "hmap_generator.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>


static void print_usage(const char *prog) {
    std::cout << "Usage: " << prog
              << " -c context -p precision -m model_path (-i image_path | -d input_dir -o output_dir)"
              << std::endl;
}

static void mkdir_p(const std::string &file_path) {
    if (file_path.empty()) {
        return;
    }
    size_t end = file_path.find_last_of('/');
    if (end == std::string::npos) {
        return;
    }
    std::string dir = file_path.substr(0, end);
    size_t pos = 0;
    while ((pos = dir.find('/', pos + 1)) != std::string::npos) {
        std::string sub = dir.substr(0, pos);
        if (!sub.empty()) {
            mkdir(sub.c_str(), 0755);
        }
    }
    mkdir(dir.c_str(), 0755);
}

static bool has_image_ext(const std::string &path) {
    const auto dot = path.rfind('.');
    if (dot == std::string::npos || dot + 1 >= path.size()) {
        return false;
    }
    std::string ext = path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "bmp";
}

static bool should_skip_path(const std::string &path) {
    return path.find("/._") != std::string::npos ||
           path.find("__MACOSX") != std::string::npos ||
           path.find("/.DS_Store") != std::string::npos;
}

static void walk_images(const std::string &dir, std::vector<std::string> &images) {
    DIR *dp = opendir(dir.c_str());
    if (dp == nullptr) {
        return;
    }

    dirent *entry = nullptr;
    while ((entry = readdir(dp)) != nullptr) {
        if (entry->d_name[0] == '.') {
            continue;
        }

        const std::string path = dir + "/" + entry->d_name;
        struct stat st {};
        if (stat(path.c_str(), &st) != 0) {
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            walk_images(path, images);
            continue;
        }

        if (!S_ISREG(st.st_mode) || !has_image_ext(path) || should_skip_path(path)) {
            continue;
        }
        images.push_back(path);
    }
    closedir(dp);
}

static std::string normalize_dir(const std::string &dir) {
    if (dir.size() > 1 && dir.back() == '/') {
        return dir.substr(0, dir.size() - 1);
    }
    return dir;
}

static void collect_images(const std::string &input_dir, std::vector<std::string> &images) {
    walk_images(normalize_dir(input_dir), images);
    std::sort(images.begin(), images.end());
    images.erase(std::unique(images.begin(), images.end()), images.end());
}

static bool infer_one(HeatMapGenerator &hmap_generator,
                      const std::string &image_path,
                      const std::string &blend_path,
                      const std::string &heatmap_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        fprintf(stderr, "Failed to read image: %s\n", image_path.c_str());
        return false;
    }

    cv::Mat heatmap = hmap_generator.Infer(image);
    Hotspot max_hotspot = hmap_generator.LocateMaxHotspot(heatmap);
    printf("max_hot intensity: %f (%s)\n", max_hotspot.mean_intensity, image_path.c_str());

    cv::Mat blend_img;
    cv::Mat image_bgr;
    cv::Mat heatmap_bgr;
    cv::cvtColor(image, image_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(heatmap, heatmap_bgr, cv::COLOR_RGB2BGR);
    cv::addWeighted(image_bgr, 0.5, heatmap_bgr, 0.5, 0, blend_img);

    mkdir_p(blend_path);
    mkdir_p(heatmap_path);
    if (!cv::imwrite(blend_path, blend_img)) {
        fprintf(stderr, "Failed to write blend image: %s\n", blend_path.c_str());
        return false;
    }
    if (!cv::imwrite(heatmap_path, heatmap)) {
        fprintf(stderr, "Failed to write heatmap image: %s\n", heatmap_path.c_str());
        return false;
    }
    return true;
}

static std::string relative_path(const std::string &root, const std::string &path) {
    if (path.size() <= root.size()) {
        return path;
    }
    if (path.compare(0, root.size(), root) != 0) {
        return path;
    }
    std::string rel = path.substr(root.size());
    if (!rel.empty() && rel[0] == '/') {
        rel = rel.substr(1);
    }
    return rel;
}

static std::string strip_extension(const std::string &path) {
    const auto dot = path.rfind('.');
    if (dot == std::string::npos) {
        return path;
    }
    return path.substr(0, dot);
}

int main(int argc, char **argv) {
    int ch;
    std::string context = "cpu";
    std::string precision = "uint8";
    std::string model_path;
    std::string image_path;
    std::string input_dir;
    std::string output_dir;

    while ((ch = getopt(argc, argv, "c:p:m:i:d:o:")) != -1) {
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
            case 'd':
                input_dir = optarg;
                break;
            case 'o':
                output_dir = optarg;
                break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    const bool batch_mode = !input_dir.empty();
    if (batch_mode) {
        if (output_dir.empty() || !image_path.empty()) {
            print_usage(argv[0]);
            return 1;
        }
    } else if (image_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    HeatMapGenerator hmap_generator(context, precision);
    hmap_generator.Init(model_path);

    if (!batch_mode) {
        return infer_one(hmap_generator, image_path, "./blend.png", "./heatmap.png") ? 0 : 1;
    }

    std::vector<std::string> images;
    const std::string normalized_input_dir = normalize_dir(input_dir);
    collect_images(normalized_input_dir, images);
    if (images.empty()) {
        fprintf(stderr, "No images found under: %s\n", input_dir.c_str());
        return 1;
    }

    int processed = 0;
    for (const std::string &path : images) {
        const std::string rel = relative_path(normalized_input_dir, path);
        const std::string rel_no_ext = strip_extension(rel);
        const std::string blend_path = output_dir + "/" + rel_no_ext + ".png";
        const std::string heatmap_path = output_dir + "/" + rel_no_ext + ".heatmap.png";
        if (!infer_one(hmap_generator, path, blend_path, heatmap_path)) {
            return 1;
        }
        processed += 1;
        printf("[%d/%zu] %s\n", processed, images.size(), rel.c_str());
    }

    printf("batch_done: %d images\n", processed);
    return 0;
}
