#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <random>
#include <deque>
#include <array>

class EMGMM {
public:
    enum class ReceptionState { Blocked = 0, LOS = 1, NLOS = 2 };

    struct Config {
        int max_iterations = 100;
        double tolerance = 1e-4;
        int n_init = 3;
        double min_variance = 1.0;
    };

    struct GaussianComponent {
        double mean = 0.0;
        double variance = 1.0;
        double weight = 0.0;
        std::vector<double> points;
    };

    struct Result {
        double log_likelihood = -std::numeric_limits<double>::infinity();
        std::vector<GaussianComponent> components;
    };

    // === ВСТРОЕННОЕ СКОЛЬЗЯЩЕЕ ОКНО ===
    struct SlidingWindow {
        struct Obs { double ts; double err; double cn0; };
        std::deque<Obs> buffer;
        double duration = 40.0;

        SlidingWindow(double dur = 40.0) : duration(dur) {}

        void add(double ts, double err, double cn0) {
            buffer.push_back({ts, err, cn0});
            trim();
        }

        void updateDuration(double speed_mps) {
            if (speed_mps > 10.0) duration = 40.0;
            else if (speed_mps < 1.0) duration = 240.0;
            else duration = 40.0 + (240.0 - 40.0) * (10.0 - speed_mps) / 10.0;
            trim();
        }

        static int classifyState(double cn0, double los_thr = 40.0, double block_thr = 30.0) {
            if (cn0 < block_thr) return static_cast<int>(ReceptionState::Blocked);
            if (cn0 >= los_thr) return static_cast<int>(ReceptionState::LOS);
            return static_cast<int>(ReceptionState::NLOS);
        }

        std::vector<double> getErrors(ReceptionState state) const {
            std::vector<double> res;
            int target = static_cast<int>(state);
            for (const auto& o : buffer) {
                if (classifyState(o.cn0) == target) res.push_back(o.err);
            }
            return res;
        }

        std::array<std::array<double, 3>, 3> estimateTransitionMatrix() const {
            std::array<std::array<int, 3>, 3> counts{};
            if (buffer.size() < 2) {
                std::array<std::array<double, 3>, 3> identity{};
                identity[0][0] = 1.0; identity[1][1] = 1.0; identity[2][2] = 1.0;
                return identity;
            }

            auto prev = classifyState(buffer.front().cn0);
            for (size_t i = 1; i < buffer.size(); ++i) {
                int curr = classifyState(buffer[i].cn0);
                if (prev != static_cast<int>(ReceptionState::Blocked) || 
                    curr != static_cast<int>(ReceptionState::Blocked)) {
                    counts[prev][curr]++;
                }
                prev = curr;
            }

            std::array<std::array<double, 3>, 3> M{};
            for (int i = 0; i < 3; ++i) {
                int row = counts[i][0] + counts[i][1] + counts[i][2];
                if (row > 0) {
                    for (int j = 0; j < 3; ++j) M[i][j] = static_cast<double>(counts[i][j]) / row;
                } else {
                    M[i][i] = 1.0;
                }
            }
            return M;
        }

    private:
        void trim() {
            while (!buffer.empty() && buffer.back().ts - buffer.front().ts > duration) {
                buffer.pop_front();
            }
        }
    };

    EMGMM() : config_() {}
    explicit EMGMM(const Config& config) : config_(config) {}

    static double gaussian_pdf(double x, double mean, double var, double min_var = 1e-6);
    static double log_sum_exp(const std::vector<double>& log_vals);

    std::vector<GaussianComponent> run(const std::vector<double>& data, int max_k);
    Result fit_with_state(const std::vector<double>& data, ReceptionState state, int max_k = 2);
    Result fit(const std::vector<double>& data, int k);

private:
    Config config_;
    Result fit_single(const std::vector<double>& data, int k, const std::vector<double>& init_means = {});
    std::vector<double> initialize_centroids(const std::vector<double>& data, int k);
    std::vector<double> initialize_with_offset(const std::vector<double>& data, int k);
    int find_best_k(const std::vector<double>& data, int max_k, ReceptionState state);
    double compute_log_likelihood(const std::vector<double>& data, const std::vector<GaussianComponent>& comps);
};