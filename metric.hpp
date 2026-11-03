#pragma once
#include <vector>

class MetricsCalculator {
public:
    struct WindowMetrics {
        double time = 0.0;
        double silhouette = -1.0;
        double davies_bouldin = -1.0;
        double inertia = -1.0; 
        int    n_modes = 0;
    };

    static WindowMetrics computeForBin(
        double bin_time,
        const std::vector<double>& cleaned_errors,
        const std::vector<double>& mode_centers,
        const std::vector<double>& mode_stds,
        const std::vector<double>& mode_weights,
        const std::vector<int>& point_clusters);
};