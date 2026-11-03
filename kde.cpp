#include "kde.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

double KDE::gaussian_kernel(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

double KDE::epanechnikov_kernel(double x) {
    double u = std::abs(x);
    return (u <= 1.0) ? 0.75 * (1.0 - u * u) : 0.0;
}

double KDE::silverman_bandwidth(const std::vector<double>& data) {
    if (data.size() < 2) return 1.0;
    
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double variance = 0.0;
    
    for (double x : data) {
        variance += (x - mean) * (x - mean);
    }
    variance /= (data.size() - 1);
    
    double std_dev = std::sqrt(variance);
    return 0.9 * std_dev * std::pow(data.size(), -0.2);
}

std::vector<KDE::DensityPoint> KDE::estimate(const std::vector<double>& errors, 
                                            double bandwidth,
                                            double x_min,
                                            double x_max,
                                            int num_points) {
    if (errors.empty()) {
        throw std::invalid_argument("Данные для KDE пусты.");
    }
    
    if (bandwidth <= 0.0) {
        bandwidth = silverman_bandwidth(errors);
    }
    
    std::vector<DensityPoint> result;
    double step = (x_max - x_min) / (num_points - 1);
    
    for (int i = 0; i < num_points; ++i) {
        double x = x_min + i * step;
        double density = 0.0;
        
        for (double point : errors) {
            double u = (x - point) / bandwidth;
            density += gaussian_kernel(u);
        }
        
        density /= (errors.size() * bandwidth);
        result.push_back({x, density});
    }
    
    return result;
}