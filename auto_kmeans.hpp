#pragma once

#include <vector>

class AutoKMeans {
public:
    struct Cluster {
        std::vector<double> points;
        double centroid;
    };

    std::vector<Cluster> run(const std::vector<double>& data, int max_possible_k = 10);

private:
    std::vector<Cluster> cluster(const std::vector<double>& data, int k);
    double calculate_wcss(const std::vector<Cluster>& clusters);
    int find_best_k(const std::vector<double>& data, int max_k);
};