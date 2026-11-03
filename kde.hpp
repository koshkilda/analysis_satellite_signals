#pragma once
#include <vector>
#include <map>

class KDE {
public:
    struct DensityPoint {
        double x;
        double density;
    };
    
    static double gaussian_kernel(double x);
    static double epanechnikov_kernel(double x);
    static double silverman_bandwidth(const std::vector<double>& data);
    
    static std::vector<DensityPoint> estimate(const std::vector<double>& errors,
                                             double bandwidth,
                                             double x_min,
                                             double x_max,
                                             int num_points);
};