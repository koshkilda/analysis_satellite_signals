#ifndef ANALYSIS_UTILS_HPP
#define ANALYSIS_UTILS_HPP

#include <vector>
#include <map>
#include "kde.hpp"
#include "data_analyzer.hpp"
#include "em.hpp"

struct SatelliteData {
    double timestamp;
    double error;
};

namespace AnalysisUtils {

    constexpr int MAX_WINDOW_CLUSTERS = 3;

    struct WindowFeatures {
        double centers[MAX_WINDOW_CLUSTERS]{};
        double weights[MAX_WINDOW_CLUSTERS]{};
    };

    std::vector<int> applyDBSCAN(const std::vector<double>& data, double eps, int minPts);

    std::vector<DataAnalyzer::TimeBinStats::Mode> findModesViaEM(
        const std::vector<double>& errors,
        int max_components,
        EMGMM::ReceptionState state = EMGMM::ReceptionState::NLOS);

    std::vector<KDE::DensityPoint> estimateDensityViaKDE(
        const std::vector<double>& errors,  
        double bandwidth,                   
        int steps);                           
        
    std::vector<KDE::DensityPoint> reconstructDensityFromModes(
        const std::vector<DataAnalyzer::TimeBinStats::Mode>& modes,
        const std::vector<double>& raw_errors,
        int steps);

    std::vector<DataAnalyzer::TimeBinStats::Mode> clusterDensityWithDBSCAN(
        const std::vector<double>& errors,
        double eps, int minPts);
    
    std::vector<DataAnalyzer::TimeBinStats::Mode> clusterDensityWithAutoKMeans(
        const std::vector<double>& errors,
        int maxK);

    std::vector<DataAnalyzer::TimeBinStats::Mode> stabilizeModesTracking(
        const std::vector<DataAnalyzer::TimeBinStats::Mode>& current_modes,
        const std::vector<DataAnalyzer::TimeBinStats::Mode>& previous_modes);

} // namespace AnalysisUtils

#endif // ANALYSIS_UTILS_HPP