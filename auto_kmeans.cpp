#include "auto_kmeans.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>

std::vector<AutoKMeans::Cluster> AutoKMeans::run(const std::vector<double>& data, int max_possible_k) {
    if (data.empty()) return {};

    int best_k = find_best_k(data, max_possible_k);
    return cluster(data, best_k);
}

std::vector<AutoKMeans::Cluster> AutoKMeans::cluster(const std::vector<double>& data, int k) {
    int max_iterations = 100;
    std::vector<Cluster> clusters(k);
    std::vector<double> centroids(k);
    std::vector<int> assignments(data.size(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);
    centroids[0] = data[dis(gen)];

    for (int i = 1; i < k; ++i) {
        std::vector<double> distances(data.size());
        double total_distance = 0.0;
        for (size_t j = 0; j < data.size(); ++j) {
            double min_dist = std::numeric_limits<double>::max();
            for (int c = 0; c < i; ++c) {
                double dist = std::abs(data[j] - centroids[c]);
                min_dist = std::min(min_dist, dist);
            }
            distances[j] = min_dist * min_dist;
            total_distance += distances[j];
        }

        std::uniform_real_distribution<> dis_real(0.0, total_distance);
        double rand_val = dis_real(gen);
        double cumulative = 0.0;
        for (size_t j = 0; j < data.size(); ++j) {
            cumulative += distances[j];
            if (cumulative >= rand_val) {
                centroids[i] = data[j];
                break;
            }
        }
    }

    for (int iter = 0; iter < max_iterations; ++iter) {
        bool changed = false;
        for (size_t i = 0; i < data.size(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int closest = 0;
            for (int c = 0; c < k; ++c) {
                double dist = std::abs(data[i] - centroids[c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest = c;
                }
            }
            if (assignments[i] != closest) {
                assignments[i] = closest;
                changed = true;
            }
        }

        for (int c = 0; c < k; ++c) clusters[c].points.clear();
        for (size_t i = 0; i < data.size(); ++i) {
            clusters[assignments[i]].points.push_back(data[i]);
        }

        for (int c = 0; c < k; ++c) {
            if (!clusters[c].points.empty()) {
                double sum = std::accumulate(clusters[c].points.begin(), clusters[c].points.end(), 0.0);
                centroids[c] = sum / clusters[c].points.size();
                clusters[c].centroid = centroids[c];
            }
        }
        if (!changed) break;
    }
    return clusters;
}

double AutoKMeans::calculate_wcss(const std::vector<Cluster>& clusters) {
    double wcss = 0.0;
    for (const auto& cluster : clusters) {
        for (double point : cluster.points) {
            double diff = point - cluster.centroid;
            wcss += diff * diff;
        }
    }
    return wcss;
}

int AutoKMeans::find_best_k(const std::vector<double>& data, int max_k) {
    max_k = std::min(max_k, (int)data.size());
    if (max_k < 2) return 1;

    std::vector<double> wcss_values;
    
    for (int k = 1; k <= max_k; ++k) {
        auto clusters = cluster(data, k);
        wcss_values.push_back(calculate_wcss(clusters));
    }

    int n = wcss_values.size();
    if (n < 3) return 1;

    double x0 = 1.0, y0 = wcss_values[0];
    double x1 = (double)n, y1 = wcss_values[n - 1];
    
    double max_dist = -1.0;
    int best_k = 1;

    for (int i = 1; i < n - 1; ++i) {
        double xi = (double)(i + 1);
        double yi = wcss_values[i];
        
        double numerator = std::abs((y1 - y0) * xi - (x1 - x0) * yi + x1 * y0 - y1 * x0);
        double denominator = std::sqrt(std::pow(y1 - y0, 2) + std::pow(x1 - x0, 2));
        double dist = numerator / denominator;

        if (dist > max_dist) {
            max_dist = dist;
            best_k = i + 1;
        }
    }

    return best_k;
}