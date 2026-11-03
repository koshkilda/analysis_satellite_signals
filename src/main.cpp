#include <QCoreApplication>
#include <QDebug>
#include <QProcess> 
#include <QDir>
#include <QFileInfo>
#include <QFile>
#include <QTextStream>
#include "data_analyzer.hpp"
#include "metric.hpp"

// ============================================================================
// КОНФИГУРАЦИЯ
// ============================================================================
struct AppConfig {
    QString input_file;
    QString output_dir = "results";
    BinningConfig bin{120.0};
    ModeFinderConfig mode;
    FilterConfig filter;
    
    AppConfig() {
        mode.method = ModeFinderMethod::EM_GMM;
        mode.em_max_components = 3;
        mode.kde_bandwidth = 0.4;
        mode.kde_steps = 400;
        mode.cluster_eps = 0.4;
        mode.cluster_minPts = 3;
        mode.autokmeans_max_k = 3;
        
        filter.enabled = false;
        filter.eps = 3.0;
        filter.minPts = 3;
    }
};

// ============================================================================
// ПАРСИНГ МЕТОДА
// ============================================================================
static bool ParseMethod(const QString& val, ModeFinderMethod& out) {
    QString v = val.toLower().trimmed();
    if (v == "em_gmm" || v == "em-gmm" || v == "em") { 
        out = ModeFinderMethod::EM_GMM; 
        return true; 
    }
    if (v == "kde_dbscan" || v == "kde-dbscan" || v == "dbscan") { 
        out = ModeFinderMethod::KDE_DBSCAN; 
        return true; 
    }
    if (v == "kde_kmeans" || v == "kde-kmeans" || v == "autokmeans" || v == "kmeans") { 
        out = ModeFinderMethod::KDE_AutoKMeans; 
        return true; 
    }
    return false;
}

// ============================================================================
// СБОРКА КОНФИГА
// ============================================================================
static bool BuildConfigFromCLI(int argc, char* argv[], AppConfig& cfg) {
    auto arg = [argv](int i) { return QString::fromLocal8Bit(argv[i]); };
    
    if (argc < 2) return false;
    
    cfg.input_file = QFileInfo(arg(1)).absoluteFilePath();
    if (!QFile::exists(cfg.input_file)) return false;
    
    for (int i = 2; i < argc; ++i) {
        QString key = arg(i);
        
        if (key == "--output-dir" && i + 1 < argc) {
            cfg.output_dir = QFileInfo(arg(++i)).absoluteFilePath();
        }
        else if (key == "--bin-width" && i + 1 < argc) {
            cfg.bin.bin_width_sec = arg(++i).toDouble();
        }
        else if (key == "--em-max-comp" && i + 1 < argc) {
            cfg.mode.em_max_components = arg(++i).toInt();
        }
        else if (key == "--kde-bandwidth" && i + 1 < argc) {
            cfg.mode.kde_bandwidth = arg(++i).toDouble();
        }
        else if (key == "--kde-steps" && i + 1 < argc) {
            cfg.mode.kde_steps = arg(++i).toInt();
        }
        else if (key == "--cluster-eps" && i + 1 < argc) {
            cfg.mode.cluster_eps = arg(++i).toDouble();
        }
        else if (key == "--cluster-minpts" && i + 1 < argc) {
            cfg.mode.cluster_minPts = arg(++i).toInt();
        }
        else if (key == "--cluster-max-k" && i + 1 < argc) {
            cfg.mode.autokmeans_max_k = arg(++i).toInt();
        }
        else if (key == "--method" && i + 1 < argc) {
            if (!ParseMethod(arg(++i), cfg.mode.method)) {
                qWarning() << "Unknown method, using default (EM_GMM)";
            }
        }
        else if (key == "--filter") {
            cfg.filter.enabled = true;
        }
        else if (key == "--filter-eps" && i + 1 < argc) {
            cfg.filter.eps = arg(++i).toDouble();
        }
        else if (key == "--filter-minpts" && i + 1 < argc) {
            cfg.filter.minPts = arg(++i).toInt();
        }
    }
    return true;
}

// ============================================================================
// ЭКСПОРТ МЕТРИК
// ============================================================================
static bool exportMetricsToCSV(const std::vector<MetricsCalculator::WindowMetrics>& metrics,
                               const QString& filename) {
    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly)) return false;
    QTextStream out(&file);
    
    out << "time,silhouette,davies_bouldin,inertia,n_modes\n";
    for (const auto& m : metrics) {
        out << m.time << "," << m.silhouette << "," << m.davies_bouldin << ","
            << m.inertia << "," << m.n_modes << "\n";
    }
    return true;
}

// ============================================================================
// ЗАПУСК АНАЛИЗА
// ============================================================================
static bool RunAnalysis(const AppConfig& cfg) {
    qDebug() << "=== Analyzer ===";
    qDebug() << "Input:" << cfg.input_file;
    QDir dir(cfg.output_dir);
    if (!dir.exists() && !dir.mkpath(".")) {
        qCritical() << "Cannot create output dir:" << cfg.output_dir;
        return false;
    }

    DataAnalyzer analyzer;
    auto data = analyzer.read_csv_qt(cfg.input_file);
    if (data.isEmpty()) { qCritical() << "No data"; return false; }

    auto stats = analyzer.computeTimeBinStats(data, cfg.bin, cfg.mode, cfg.filter);
    if (stats.empty()) { qCritical() << "No stats"; return false; }

    // Экспорт unified CSV
    QString out_unified = dir.absoluteFilePath("data_unified.csv");
    if (!analyzer.exportUnifiedDataToCSV(cfg.input_file, out_unified.toStdString(),
                                         cfg.mode, cfg.filter, cfg.bin, nullptr)) {
        qCritical() << "Unified export failed";
        return false;
    }

    // Вычисляем метрики
    std::vector<MetricsCalculator::WindowMetrics> all_metrics;
    for (const auto& bin : stats) {
        auto clusters = DataAnalyzer::assignClustersToPoints(bin.errors, bin.modes);
        std::vector<double> centers, stds, weights;
        for (const auto& m : bin.modes) {
            centers.push_back(m.error);
            stds.push_back(m.std_error);
            weights.push_back(m.density);
        }
        auto bin_metrics = MetricsCalculator::computeForBin(bin.time, bin.errors, centers, stds, weights, clusters);
        all_metrics.push_back(bin_metrics);
    }

    // Экспорт метрик
    QString out_metrics = dir.absoluteFilePath("metrics.csv");
    if (!exportMetricsToCSV(all_metrics, out_metrics)) {
        qWarning() << "Could not export metrics";
    } else {
        qDebug() << "Metrics saved to" << out_metrics;
    }

    // Вызов Python-скрипта для визуализации с передачей обоих CSV
    if (QFile::exists("plot_generator.py")) {
        QProcess proc;
        QStringList args;
        args << "plot_generator.py"
             << out_unified
             << "--output-dir" << cfg.output_dir
             << "--metrics-csv" << out_metrics;
        proc.start("python3", args);
        proc.waitForFinished(120000);
        if (proc.exitCode() == 0)
            qDebug() << "Plots ready in" << cfg.output_dir;
        else
            qWarning() << "Plot script failed with code" << proc.exitCode();
    } else {
        qWarning() << "plot_generator.py not found, skipping plots";
    }

    return true;
}

// ============================================================================
// ENTRY POINT
// ============================================================================
int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);
    AppConfig cfg;
    if (!BuildConfigFromCLI(argc, argv, cfg)) {
        qCritical() << "Usage:" << argv[0] << "<input.csv> [--method em|kde-dbscan|kde-kmeans] [--filter] ...";
        return 1;
    }
    return RunAnalysis(cfg) ? 0 : 1;
}