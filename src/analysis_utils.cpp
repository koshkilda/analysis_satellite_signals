#include "analysis_utils.hpp"
#include "auto_kmeans.hpp"  
#include "dbscan.hpp"
#include "em.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <map>
#include <limits>

// ============================================================================
// Вспомогательная функция: среднее и стандартное отклонение
// ============================================================================
static std::pair<double, double> mean_and_std(const std::vector<double>& values) {
    if (values.empty()) return {0.0, 0.0};
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double var = 0.0;
    for (double x : values) {
        double d = x - mean;
        var += d * d;
    }
    return {mean, std::sqrt(var / values.size())};
}

std::vector<int> AnalysisUtils::applyDBSCAN(
    const std::vector<double>& data, double eps, int minPts)
{
    if (data.empty()) return {};
    DBSCAN dbscan(eps, minPts);
    return dbscan.fit_predict(data);
}

std::vector<DataAnalyzer::TimeBinStats::Mode> 
AnalysisUtils::findModesViaEM(
    const std::vector<double>& errors, 
    int max_components,
    EMGMM::ReceptionState state)
{
    std::vector<DataAnalyzer::TimeBinStats::Mode> modes;
    if (errors.empty() || max_components < 1) return modes;

    EMGMM::Config cfg{100, 1e-4, 3, 0.12};
    EMGMM em(cfg);
    auto res = em.fit_with_state(errors, state, max_components);

    for (const auto& c : res.components) {
        if (c.weight < 0.05) continue;
        modes.push_back({c.mean, std::sqrt(c.variance), c.weight});
    }

    std::sort(modes.begin(), modes.end(),
              [](const auto& a, const auto& b) { return a.density > b.density; });
    return modes;
}

// ============================================================================
// DBSCAN НА СЫРЫХ ДАННЫХ
// ============================================================================
std::vector<DataAnalyzer::TimeBinStats::Mode> 
AnalysisUtils::clusterDensityWithDBSCAN(
    const std::vector<double>& errors,
    double eps, int minPts)
{
    std::vector<DataAnalyzer::TimeBinStats::Mode> modes;
    if (errors.empty()) return modes;
    
    auto labels = applyDBSCAN(errors, eps, minPts);
    
    std::map<int, std::vector<double>> clusters;
    for (size_t i = 0; i < errors.size(); ++i) {
        if (labels[i] != -1) {
            clusters[labels[i]].push_back(errors[i]);
        }
    }
    
    // Если шум или нет кластеров — возвращаем одну моду по всем данным
    if (clusters.empty()) {
        auto [mean, stddev] = mean_and_std(errors);
        return {{mean, stddev, 1.0}};
    }
    
    size_t total_valid = 0;
    for (const auto& c : clusters) total_valid += c.second.size();
    
    for (const auto& [label, points] : clusters) {
        if (points.empty()) continue;
        auto [mean, stddev] = mean_and_std(points);                      // REFACTOR
        double weight = static_cast<double>(points.size()) / static_cast<double>(total_valid);
        modes.push_back({mean, stddev, weight});
    }
    
    std::sort(modes.begin(), modes.end(),
              [](const auto& a, const auto& b) { return a.density > b.density; });
    return modes;
}

// ============================================================================
// КЛАСТЕРИЗАЦИЯ С ПРОВЕРКОЙ НА ОДНОМОДАЛЬНОСТЬ
// ============================================================================
std::vector<DataAnalyzer::TimeBinStats::Mode> 
AnalysisUtils::clusterDensityWithAutoKMeans(
    const std::vector<double>& errors,
    int maxK)
{
    std::vector<DataAnalyzer::TimeBinStats::Mode> modes;
    if (errors.empty() || maxK < 1) return modes;
    
    AutoKMeans akmeans;
    auto clusters = akmeans.run(errors, maxK);
    if (clusters.empty()) return modes;
    
    // Проверка на одномодальность
    size_t max_cluster_size = 0;
    for (const auto& c : clusters) {
        max_cluster_size = std::max(max_cluster_size, c.points.size());
    }
    
    bool should_merge = false;
    if (clusters.size() >= 2) {
        std::vector<std::pair<double, size_t>> sorted_centers;
        for (size_t i = 0; i < clusters.size(); ++i) {
            if (!clusters[i].points.empty()) {
                sorted_centers.push_back({clusters[i].centroid, i});
            }
        }
        std::sort(sorted_centers.begin(), sorted_centers.end());
        
        const double MIN_CLUSTER_DISTANCE = 0.8;
        for (size_t i = 1; i < sorted_centers.size(); ++i) {
            double dist = sorted_centers[i].first - sorted_centers[i-1].first;
            if (dist < MIN_CLUSTER_DISTANCE) {
                should_merge = true;
                break;
            }
        }
    }
    
    // Если режим слияния или один кластер — возвращаем глобальную моду
    if (should_merge || clusters.size() == 1) {
        auto [mean, stddev] = mean_and_std(errors);                     // REFACTOR
        return {{mean, stddev, 1.0}};
    }
    
    // Многомодальное распределение
    size_t total_points = errors.size();
    for (const auto& cluster : clusters) {
        if (cluster.points.empty()) continue;
        auto [mean, stddev] = mean_and_std(cluster.points);             // REFACTOR
        double weight = static_cast<double>(cluster.points.size()) / static_cast<double>(total_points);
        modes.push_back({mean, stddev, weight});
    }
    
    std::sort(modes.begin(), modes.end(),
              [](const auto& a, const auto& b) { return a.density > b.density; });
    return modes;
}

// Прямой непараметрический KDE для визуализации
std::vector<KDE::DensityPoint> 
AnalysisUtils::estimateDensityViaKDE(
    const std::vector<double>& errors,
    double bandwidth,
    int steps)
{
    std::vector<KDE::DensityPoint> profile;
    if (errors.empty() || steps < 1 || bandwidth <= 1e-9) return profile;   // REFACTOR: проверка bandwidth
    
    auto [minIt, maxIt] = std::minmax_element(errors.begin(), errors.end());
    double minE = *minIt, maxE = *maxIt;
    double range = maxE - minE;
    if (range < 1e-6) { minE -= 1.0; maxE += 1.0; }
    else { minE -= range * 0.2; maxE += range * 0.2; }
    
    double step_size = (maxE - minE) / static_cast<double>(steps);
    
    for (int i = 0; i <= steps; ++i) {
        double x = minE + i * step_size;
        double density = 0.0;
        for (double e : errors) {
            double u = (x - e) / bandwidth;
            density += KDE::gaussian_kernel(u);
        }
        density /= (static_cast<double>(errors.size()) * bandwidth);
        profile.push_back({x, density});
    }
    return profile;
}

// ============================================================================
// Восстановление профиля плотности из найденных Мод
// ============================================================================
std::vector<KDE::DensityPoint> 
AnalysisUtils::reconstructDensityFromModes(
    const std::vector<DataAnalyzer::TimeBinStats::Mode>& modes,
    const std::vector<double>& raw_errors,
    int steps)
{
    std::vector<KDE::DensityPoint> profile;
    if (modes.empty() || raw_errors.empty() || steps < 1) return profile;

    auto [minIt, maxIt] = std::minmax_element(raw_errors.begin(), raw_errors.end());
    double minE = *minIt, maxE = *maxIt;
    double range = maxE - minE;
    
    if (range < 1e-6) { minE -= 1.0; maxE += 1.0; }
    else { minE -= range * 0.15; maxE += range * 0.15; }

    double step_size = (maxE - minE) / static_cast<double>(steps);
    
    double total_weight = 0.0;
    for(const auto& m : modes) total_weight += m.density;
    if (total_weight < 1e-6) total_weight = 1.0;
    
    std::vector<double> norm_weights;
    norm_weights.reserve(modes.size());
    for(const auto& m : modes) {
        norm_weights.push_back(m.density / total_weight);
    }

    for (int i = 0; i <= steps; ++i) {
        double x = minE + i * step_size;
        double density_sum = 0.0;
        for (size_t j = 0; j < modes.size(); ++j) {
            const auto& mode = modes[j];
            double u = (x - mode.error) / mode.std_error;
            double pdf_val = norm_weights[j] * (1.0 / mode.std_error) * KDE::gaussian_kernel(u);
            density_sum += pdf_val;
        }
        profile.push_back({x, density_sum});
    }
    return profile;
}

// ============================================================================
// СТАБИЛИЗАЦИЯ МОД: Tracking между временными шагами
// ============================================================================
std::vector<DataAnalyzer::TimeBinStats::Mode>
AnalysisUtils::stabilizeModesTracking(
    const std::vector<DataAnalyzer::TimeBinStats::Mode>& current_modes,
    const std::vector<DataAnalyzer::TimeBinStats::Mode>& previous_modes)
{
    if (previous_modes.empty() || current_modes.size() != previous_modes.size()) {
        return current_modes;
    }
    
    const size_t n_modes = current_modes.size();
    std::vector<DataAnalyzer::TimeBinStats::Mode> stabilized(n_modes);
    std::vector<bool> assigned(n_modes, false);
    
    for (size_t old_idx = 0; old_idx < n_modes; ++old_idx) {
        double prev_center = previous_modes[old_idx].error;
        double min_dist = std::numeric_limits<double>::max();
        size_t best_new_idx = 0;
        bool found = false;
        
        for (size_t new_idx = 0; new_idx < n_modes; ++new_idx) {
            if (assigned[new_idx]) continue;
            double dist = std::abs(current_modes[new_idx].error - prev_center);
            if (dist < min_dist) {
                min_dist = dist;
                best_new_idx = new_idx;
                found = true;
            }
        }
        
        if (found) {
            stabilized[old_idx] = current_modes[best_new_idx];
            assigned[best_new_idx] = true;
        }
    }
    
    return stabilized;
}