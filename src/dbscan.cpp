#include "dbscan.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

DBSCAN::DBSCAN(double eps, int min_pts) : eps_(eps), min_pts_(min_pts), next_cluster_id_(0) {
    if (eps <= 0.0) {
        throw std::invalid_argument("eps (радиус окрестности) должен быть положительным числом.");
    }
    if (min_pts <= 0) {
        throw std::invalid_argument("min_pts (мин. кол-во точек в окрестности) должно быть положительным числом.");
    }
}

std::vector<int> DBSCAN::region_query(const std::vector<double>& data, size_t point_idx) const {
    std::vector<int> neighbors;
    double center = data[point_idx];
    
    for (size_t i = 0; i < data.size(); ++i) {
        if (i != point_idx && std::abs(data[i] - center) <= eps_) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}

void DBSCAN::expand_cluster(const std::vector<double>& data, std::vector<int>& labels, size_t point_idx, int cluster_id) {
    std::vector<int> seeds = region_query(data, point_idx);
    
    if (static_cast<int>(seeds.size()) < min_pts_) {
        labels[point_idx] = -1; 
        return;
    }

    labels[point_idx] = cluster_id;
    size_t current_idx = 0;
    
    while (current_idx < seeds.size()) {
        size_t neighbor_idx = seeds[current_idx];
        
        if (labels[neighbor_idx] == -2) { 
            labels[neighbor_idx] = cluster_id;
            
            std::vector<int> neighbor_seeds = region_query(data, neighbor_idx);
            if (static_cast<int>(neighbor_seeds.size()) >= min_pts_) {
                for (int new_idx : neighbor_seeds) {
                    bool found = false;
                    for (int existing_idx : seeds) {
                        if (existing_idx == new_idx) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        seeds.push_back(new_idx);
                    }
                }
            }
        }
        current_idx++;
    }
}

std::vector<int> DBSCAN::fit_predict(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Данные для обучения пусты.");
    }

    size_t n_points = data.size();
    std::vector<int> labels(n_points, -2); 

    for (size_t i = 0; i < n_points; ++i) {
        if (labels[i] != -2) continue; 

        std::vector<int> neighbors = region_query(data, i);

        if (static_cast<int>(neighbors.size()) < min_pts_) {
            labels[i] = -1; // Шум
        } else {
            expand_cluster(data, labels, i, next_cluster_id_);
            next_cluster_id_++;
        }
    }
    return labels;
}

std::vector<DBSCAN::Cluster> DBSCAN::cluster(const std::vector<double>& errors, double eps, int min_points) {
    DBSCAN dbscan(eps, min_points);
    auto labels = dbscan.fit_predict(errors);
    
    std::map<int, std::vector<double>> cluster_map;
    std::vector<double> noise_points;
    
    for (size_t i = 0; i < errors.size(); ++i) {
        if (labels[i] == -1) {
            noise_points.push_back(errors[i]);
        } else if (labels[i] >= 0) {
            cluster_map[labels[i]].push_back(errors[i]);
        }
    }
    
    std::vector<Cluster> clusters;
    
    for (const auto& pair : cluster_map) {
        Cluster cluster;
        cluster.points = pair.second;
        clusters.push_back(cluster);
    }
    
    if (!noise_points.empty()) {
        Cluster noise_cluster;
        noise_cluster.points = noise_points;
        noise_cluster.is_noise = true;
        clusters.push_back(noise_cluster);
    }
    
    return clusters;
}