#include "data_analyzer.hpp"
#include "analysis_utils.hpp"
#include "em.hpp"
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <algorithm>
#include <map>
#include <numeric>
#include <cmath>
#include <limits>

std::pair<double, double> DataAnalyzer::getLocalErrorRange(
    const std::vector<double>& errors, double padding_ratio)
{
    if (errors.empty()) return {0.0, 1.0};
    auto [minIt, maxIt] = std::minmax_element(errors.begin(), errors.end());
    double minE = *minIt, maxE = *maxIt;
    double range = maxE - minE;
    if (range < 1e-6) return {minE - 1.0, maxE + 1.0};
    double pad = range * padding_ratio;
    return {minE - pad, maxE + pad};
}

std::vector<int> DataAnalyzer::assignClustersToPoints(
    const std::vector<double>& errors,
    const std::vector<TimeBinStats::Mode>& modes)
{
    std::vector<int> labels(errors.size(), 0);
    if (modes.empty() || errors.empty()) return labels;
    if (modes.size() == 1) return labels;
    for (size_t i = 0; i < errors.size(); ++i) {
        double minDist = std::numeric_limits<double>::max();
        int bestCluster = 0;
        for (size_t m = 0; m < modes.size(); ++m) {
            double dist = std::abs(errors[i] - modes[m].error);
            if (dist < minDist) {
                minDist = dist;
                bestCluster = static_cast<int>(m);
            }
        }
        labels[i] = bestCluster;
    }
    return labels;
}

QVector<DataAnalyzer::PseudoRangeData> DataAnalyzer::read_csv_qt(const QString& filename) {
    QVector<PseudoRangeData> data;
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return data;
    QTextStream in(&file);
    in.readLine();
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty()) continue;
        QStringList cols = line.split(',');
        if (cols.size() != 3) continue;
        bool okT, okE;
        double t = cols[0].toDouble(&okT);
        double e = cols[2].toDouble(&okE);
        if (!okT || !okE) continue;
        data.append({t, cols[1].trimmed(), 0.0, 0.0, e});
    }
    return data;
}

std::vector<DataAnalyzer::TimeBinStats> 
DataAnalyzer::computeTimeBinStats(
    const QVector<PseudoRangeData>& data,
    const BinningConfig& binCfg,
    const ModeFinderConfig& modeCfg,
    const FilterConfig& filterCfg)
{
    if (data.isEmpty()) return {};
    std::map<int, std::vector<double>> bins;
    for (const auto& e : data) {
        int key = static_cast<int>(std::floor(e.time / binCfg.bin_width_sec) * binCfg.bin_width_sec);
        bins[key].push_back(e.error);
    }
    std::vector<TimeBinStats> result;
    result.reserve(bins.size());
    std::vector<TimeBinStats::Mode> prev_modes;
    for (const auto& [binTime, errors] : bins) {
        if (errors.empty()) continue;
        TimeBinStats stats;
        stats.time = static_cast<double>(binTime);
        std::vector<double> clean = errors;
        if (filterCfg.enabled) {
            auto labels = AnalysisUtils::applyDBSCAN(errors, filterCfg.eps, filterCfg.minPts);
            clean.clear();
            for (size_t i = 0; i < errors.size(); ++i)
                if (labels[i] != -1) clean.push_back(errors[i]);
            if (clean.empty()) clean = errors;
        }
        stats.errors = clean;
        if (!clean.empty()) {
            stats.mean_error = std::accumulate(clean.begin(), clean.end(), 0.0) / clean.size();
            double var = 0;
            for (double x : clean) { double d = x - stats.mean_error; var += d*d; }
            stats.std_error = std::sqrt(var / clean.size());
        }
        switch (modeCfg.method) {
            case ModeFinderMethod::EM_GMM:
                if (clean.size() >= 3) {
                    stats.modes = AnalysisUtils::findModesViaEM(clean, modeCfg.em_max_components);
                    stats.modes = AnalysisUtils::stabilizeModesTracking(stats.modes, prev_modes);
                    stats.density_profile = AnalysisUtils::reconstructDensityFromModes(stats.modes, clean, modeCfg.kde_steps);
                }
                break;
            case ModeFinderMethod::KDE_DBSCAN:
                if (clean.size() >= 5) {
                    stats.modes = AnalysisUtils::clusterDensityWithDBSCAN(clean, modeCfg.cluster_eps, modeCfg.cluster_minPts);
                    stats.modes = AnalysisUtils::stabilizeModesTracking(stats.modes, prev_modes);
                    stats.density_profile = AnalysisUtils::estimateDensityViaKDE(clean, modeCfg.kde_bandwidth, modeCfg.kde_steps);
                }
                break;
            case ModeFinderMethod::KDE_AutoKMeans:
                if (clean.size() >= 5) {
                    stats.modes = AnalysisUtils::clusterDensityWithAutoKMeans(clean, modeCfg.autokmeans_max_k);
                    stats.modes = AnalysisUtils::stabilizeModesTracking(stats.modes, prev_modes);
                    stats.density_profile = AnalysisUtils::estimateDensityViaKDE(clean, modeCfg.kde_bandwidth, modeCfg.kde_steps);
                }
                break;
            default:
                qCritical() << "[Stats] Unknown method:" << static_cast<int>(modeCfg.method);
                return {};
        }
        if (!stats.modes.empty()) prev_modes = stats.modes;
        result.push_back(std::move(stats));
    }
    return result;
}

bool DataAnalyzer::exportUnifiedDataToCSV(
    const QString& input_file,
    const std::string& output_csv,
    const ModeFinderConfig& modeCfg,
    const FilterConfig& filterCfg,
    const BinningConfig& binCfg,
    const std::vector<int>* referenceLabels)
{
    auto data = read_csv_qt(input_file);
    if (data.isEmpty()) { qWarning() << "No data"; return false; }
    auto stats = computeTimeBinStats(data, binCfg, modeCfg, filterCfg);
    if (stats.empty()) { qWarning() << "No stats"; return false; }
    QFile file(QString::fromStdString(output_csv));
    if (!file.open(QIODevice::WriteOnly)) { qWarning() << "Cannot open file"; return false; }
    QTextStream out(&file);
    double minTime = stats.front().time, maxTime = stats.back().time;
    double minErr = std::numeric_limits<double>::max(), maxErr = -std::numeric_limits<double>::max();
    for (const auto& s : stats) {
        if (!s.errors.empty()) {
            auto [minE, maxE] = getLocalErrorRange(s.errors, 0.0);
            minErr = std::min(minErr, minE);
            maxErr = std::max(maxErr, maxE);
        }
    }
    out << QString("#minTime=%1,maxTime=%2,minError=%3,maxError=%4\n")
           .arg(minTime).arg(maxTime).arg(minErr).arg(maxErr);
    out << "time,error,density,window_cluster,reference_mode\n";
    size_t refIdx = 0;
    for (const auto& s : stats) {
        if (s.errors.empty()) continue;
        auto pointClusters = assignClustersToPoints(s.errors, s.modes);
        for (size_t i = 0; i < s.errors.size(); ++i) {
            double t = s.time, e = s.errors[i], d = 0.0;
            if (!s.density_profile.empty()) {
                auto it = std::min_element(s.density_profile.begin(), s.density_profile.end(),
                    [e](const KDE::DensityPoint& a, const KDE::DensityPoint& b) {
                        return std::abs(a.x - e) < std::abs(b.x - e);
                    });
                if (it != s.density_profile.end()) d = it->density;
            }
            int cluster = pointClusters[i];
            int refMode = (referenceLabels && refIdx < referenceLabels->size()) ? (*referenceLabels)[refIdx] : -1;
            ++refIdx;
            out << t << "," << e << "," << d << "," << cluster << "," << refMode << "\n";
        }
    }
    return true;
}