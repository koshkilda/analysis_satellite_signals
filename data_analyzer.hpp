#ifndef DATA_ANALYZER_HPP
#define DATA_ANALYZER_HPP

#include <QString>
#include <QVector>
#include <vector>
#include "kde.hpp"

enum class ModeFinderMethod { EM_GMM, KDE_DBSCAN, KDE_AutoKMeans };

struct ModeFinderConfig {
    ModeFinderMethod method;
    int em_max_components;
    double kde_bandwidth;
    int kde_steps;
    double cluster_eps;
    int cluster_minPts;
    int autokmeans_max_k;
};

struct FilterConfig {
    bool enabled = false;
    double eps = 2.0;
    int minPts = 5;
};

struct BinningConfig {
    double bin_width_sec = 120.0;
};

class DataAnalyzer {
public:
    struct PseudoRangeData {
        double time = 0.0;
        QString sat_id;
        double pseudo_range = 0.0;
        double true_pseudo_range = 0.0;
        double error = 0.0;
    };

    struct TimeBinStats {
        double time = 0.0;
        double mean_error = 0.0;
        double std_error = 0.0;
        std::vector<double> errors;
        
        struct Mode {
            double error = 0.0;
            double std_error = 0.0;
            double density = 0.0;
        };
        std::vector<Mode> modes;
        std::vector<KDE::DensityPoint> density_profile;
    };

    // Чтение данных
    QVector<PseudoRangeData> read_csv_qt(const QString& filename);

    // Анализ (биннинг, фильтрация, поиск мод)
    std::vector<TimeBinStats> computeTimeBinStats(
        const QVector<PseudoRangeData>& data,
        const BinningConfig& binCfg,
        const ModeFinderConfig& modeCfg,
        const FilterConfig& filterCfg);

    // Назначение точек к модам (статический публичный метод)
    static std::vector<int> assignClustersToPoints(
        const std::vector<double>& errors,
        const std::vector<TimeBinStats::Mode>& modes);

    // Экспорт unified CSV
    bool exportUnifiedDataToCSV(
        const QString& input_file,
        const std::string& output_csv,
        const ModeFinderConfig& modeCfg,
        const FilterConfig& filterCfg,
        const BinningConfig& binCfg = BinningConfig(),
        const std::vector<int>* referenceLabels = nullptr);

private:
    static std::pair<double, double> getLocalErrorRange(
        const std::vector<double>& errors, double padding_ratio = 0.1);
};

#endif // DATA_ANALYZER_HPP