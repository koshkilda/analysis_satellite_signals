#include "em.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>

double EMGMM::gaussian_pdf(double x, double mean, double var, double min_var) {
    if (var < min_var) var = min_var;
    double coeff = 1.0 / std::sqrt(2.0 * M_PI * var);
    double exponent = -0.5 * std::pow(x - mean, 2) / var;
    return coeff * std::exp(exponent);
}

double EMGMM::log_sum_exp(const std::vector<double>& log_vals) {
    if (log_vals.empty()) return -std::numeric_limits<double>::infinity();
    double max_val = *std::max_element(log_vals.begin(), log_vals.end());
    if (!std::isfinite(max_val)) return max_val;
    double sum = 0.0;
    for (double lv : log_vals) sum += std::exp(lv - max_val);
    return max_val + std::log(sum + 1e-300);
}

std::vector<double> EMGMM::initialize_centroids(const std::vector<double>& data, int k) {
    if (data.empty() || k < 1) return {};
    std::vector<double> centroids;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, static_cast<int>(data.size()) - 1);
    centroids.push_back(data[dis(gen)]);

    for (int c = 1; c < k; ++c) {
        std::vector<double> weights(data.size(), 0.0);
        double total = 0.0;
        for (size_t i = 0; i < data.size(); ++i) {
            double min_d2 = std::numeric_limits<double>::max();
            for (double cent : centroids) {
                double d = data[i] - cent;
                min_d2 = std::min(min_d2, d * d);
            }
            weights[i] = min_d2;
            total += min_d2;
        }
        if (total < 1e-10) { centroids.push_back(data[dis(gen)]); continue; }
        std::uniform_real_distribution<> dis_r(0.0, total);
        double r = dis_r(gen), cum = 0.0;
        for (size_t i = 0; i < data.size(); ++i) {
            cum += weights[i];
            if (cum >= r) { centroids.push_back(data[i]); break; }
        }
    }
    return centroids;
}

std::vector<double> EMGMM::initialize_with_offset(const std::vector<double>& data, int k) {
    if (data.empty() || k < 1) return {};
    
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    
    std::vector<double> means;
    double min_val = sorted.front();
    double max_val = sorted.back();
    double step = (max_val - min_val) / std::max(1, k - 1);
    
    for (int i = 0; i < k; ++i) {
        means.push_back(min_val + i * step);
    }
    return means;
}

EMGMM::Result EMGMM::fit_with_state(const std::vector<double>& data, ReceptionState state, int max_k) {
    Result res;
    if (data.size() < 10) return res;
    if (state == ReceptionState::Blocked) return res;

    if (state == ReceptionState::LOS) {
        double m = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        if (std::abs(m) > 2.0) m = 0.0;
        double v = 0.0;
        for (double x : data) v += std::pow(x - m, 2);
        v = std::max(v / data.size(), config_.min_variance);
        
        GaussianComponent comp;
        comp.mean = m; comp.variance = v; comp.weight = 1.0;
        res.components.push_back(comp);
        res.log_likelihood = compute_log_likelihood(data, res.components);
        return res;
    }

    int k = find_best_k(data, max_k, state);
    auto init = initialize_with_offset(data, k);
    return fit_single(data, k, init);
}

EMGMM::Result EMGMM::fit_single(const std::vector<double>& data, int k, const std::vector<double>& init_means) {
    const size_t n = data.size();
    Result result;
    std::vector<double> means = init_means.empty() ? initialize_centroids(data, k) : init_means;
    std::vector<double> variances(k, 1.0);
    std::vector<double> weights(k, 1.0 / k);

    double dm = std::accumulate(data.begin(), data.end(), 0.0) / n;
    double dv = 0.0;
    for (double x : data) dv += std::pow(x - dm, 2);
    double init_var = std::max(dv / n, config_.min_variance);
    std::fill(variances.begin(), variances.end(), init_var);

    double prev_ll = -std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        std::vector<std::vector<double>> resp(n, std::vector<double>(k));
        std::vector<double> log_liks(n);

        for (size_t i = 0; i < n; ++i) {
            std::vector<double> log_probs(k);
            for (int j = 0; j < k; ++j) {
                double lp = -0.5 * std::log(2.0 * M_PI * variances[j]) 
                          - 0.5 * std::pow(data[i] - means[j], 2) / variances[j];
                log_probs[j] = lp + std::log(std::max(weights[j], 1e-300));
            }
            log_liks[i] = log_sum_exp(log_probs);
            for (int j = 0; j < k; ++j) {
                resp[i][j] = std::exp(log_probs[j] - log_liks[i]);
            }
        }

        double curr_ll = std::accumulate(log_liks.begin(), log_liks.end(), 0.0);
        if (iter > 0 && std::abs(curr_ll - prev_ll) < config_.tolerance) break;
        prev_ll = curr_ll;

        std::vector<double> Nk(k, 0.0);
        for (int j = 0; j < k; ++j)
            for (size_t i = 0; i < n; ++i) Nk[j] += resp[i][j];

        for (int j = 0; j < k; ++j) {
            if (Nk[j] < 1e-10) continue;
            weights[j] = Nk[j] / n;
            double nm = 0.0, nv = 0.0;
            for (size_t i = 0; i < n; ++i) {
                nm += resp[i][j] * data[i];
                double d = data[i] - means[j];
                nv += resp[i][j] * d * d;
            }
            means[j] = nm / Nk[j];
            variances[j] = std::max(nv / Nk[j], config_.min_variance);
        }
        double ws = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (ws > 1e-10) for (auto& w : weights) w /= ws;
    }

    result.log_likelihood = prev_ll;
    for (int j = 0; j < k; ++j) {
        GaussianComponent comp;
        comp.mean = means[j]; comp.variance = variances[j]; comp.weight = weights[j];
        result.components.push_back(comp);
    }

    for (size_t i = 0; i < n; ++i) {
        int best = 0; double mx = -1.0;
        for (int j = 0; j < k; ++j) {
            double p = gaussian_pdf(data[i], means[j], variances[j], config_.min_variance) * weights[j];
            if (p > mx) { mx = p; best = j; }
        }
        result.components[best].points.push_back(data[i]);
    }

    std::sort(result.components.begin(), result.components.end(),
              [](const GaussianComponent& a, const GaussianComponent& b) { return a.weight > b.weight; });
    return result;
}

int EMGMM::find_best_k(const std::vector<double>& data, int max_k, ReceptionState state) {
    if (state == ReceptionState::LOS) return 1;
    int eff_max = std::min(max_k, 4);
    int best_k = 1; double min_bic = std::numeric_limits<double>::max();
    int n = static_cast<int>(data.size());
    for (int k = 1; k <= eff_max; ++k) {
        Result res = fit_single(data, k);
        double bic = -2.0 * res.log_likelihood + (3 * k - 1) * std::log(n);
        if (bic < min_bic) { min_bic = bic; best_k = k; }
    }
    return best_k;
}

double EMGMM::compute_log_likelihood(const std::vector<double>& data, const std::vector<GaussianComponent>& comps) {
    double ll = 0.0;
    for (double x : data) {
        std::vector<double> lp;
        for (const auto& c : comps) {
            if (c.weight < 1e-10) continue;
            lp.push_back(-0.5*std::log(2.0*M_PI*c.variance) - 0.5*std::pow(x-c.mean,2)/c.variance + std::log(c.weight));
        }
        ll += log_sum_exp(lp);
    }
    return ll;
}

std::vector<EMGMM::GaussianComponent> EMGMM::run(const std::vector<double>& data, int max_k) {
    if (data.empty() || max_k < 1) return {};
    if (max_k > static_cast<int>(data.size()) / 3) max_k = std::max(1, static_cast<int>(data.size()) / 3);
    return fit_with_state(data, ReceptionState::NLOS, max_k).components;
}

EMGMM::Result EMGMM::fit(const std::vector<double>& data, int k) {
    Result best; double best_ll = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < config_.n_init; ++i) {
        auto cur = fit_single(data, k);
        if (cur.log_likelihood > best_ll) { best_ll = cur.log_likelihood; best = std::move(cur); }
    }
    return best;
}