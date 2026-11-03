#include "metric.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <limits>

static double silhouette_score(const std::vector<double>& errors,
                               const std::vector<int>& labels) {
    std::set<int> unique_labels(labels.begin(), labels.end());
    if (unique_labels.size() < 2) return -1.0;
    int n = errors.size();
    std::vector<double> vals;
    for (int i = 0; i < n; ++i) {
        int li = labels[i];
        if (li == -1) continue;
        double sum_same = 0.0, count_same = 0.0;
        for (int j = 0; j < n; ++j) {
            if (labels[j] == li && j != i) {
                sum_same += std::abs(errors[i] - errors[j]);
                ++count_same;
            }
        }
        double a_i = (count_same > 0) ? sum_same / count_same : 0.0;
        std::map<int, double> sum_other;
        std::map<int, int> count_other;
        for (int j = 0; j < n; ++j) {
            int lj = labels[j];
            if (lj != li && lj != -1) {
                sum_other[lj] += std::abs(errors[i] - errors[j]);
                ++count_other[lj];
            }
        }
        double b_i = std::numeric_limits<double>::max();
        for (const auto& [lbl, sum] : sum_other) {
            double avg = sum / count_other[lbl];
            if (avg < b_i) b_i = avg;
        }
        if (b_i == std::numeric_limits<double>::max()) continue;
        double s_i = (b_i - a_i) / std::max(a_i, b_i);
        vals.push_back(s_i);
    }
    if (vals.empty()) return -1.0;
    return std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
}

static double davies_bouldin_index(const std::vector<double>& /*errors*/,
                                   const std::vector<int>& /*labels*/,
                                   const std::vector<double>& centers,
                                   const std::vector<double>& stds) {
    int k = centers.size();
    if (k < 2) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        double max_ratio = 0.0;
        for (int j = 0; j < k; ++j) {
            if (i == j) continue;
            double dist = std::abs(centers[i] - centers[j]);
            if (dist < 1e-9) continue;
            double ratio = (stds[i] + stds[j]) / dist;
            if (ratio > max_ratio) max_ratio = ratio;
        }
        sum += max_ratio;
    }
    return sum / k;
}

// ============================================================================
// РАСЧЕТ ИНЕРЦИИ
// ============================================================================
static double inertia(const std::vector<double>& errors,
                      const std::vector<int>& labels,
                      const std::vector<double>& centers) {
    double inert = 0.0;
    for (size_t i = 0; i < errors.size(); ++i) {
        int lbl = labels[i];
        if (lbl >= 0 && lbl < (int)centers.size()) {
            double d = errors[i] - centers[lbl];
            inert += d * d;
        }
    }
    return inert;
}

MetricsCalculator::WindowMetrics MetricsCalculator::computeForBin(
    double bin_time,
    const std::vector<double>& cleaned_errors,
    const std::vector<double>& mode_centers,
    const std::vector<double>& mode_stds,
    const std::vector<double>& /*mode_weights*/,
    const std::vector<int>& point_clusters)
{
    WindowMetrics wm;
    wm.time = bin_time;
    wm.n_modes = mode_centers.size();
    if (cleaned_errors.empty() || mode_centers.empty() || point_clusters.empty()) {
        return wm;
    }
    wm.silhouette = silhouette_score(cleaned_errors, point_clusters);
    wm.davies_bouldin = davies_bouldin_index(cleaned_errors, point_clusters, mode_centers, mode_stds);  
    wm.inertia = inertia(cleaned_errors, point_clusters, mode_centers); 

    return wm;
}