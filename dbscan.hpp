#pragma once
#include <vector>
#include <map>

class DBSCAN {
public:
    struct Cluster {
        std::vector<double> points;
        bool is_noise = false; 
    };
    
    DBSCAN(double eps, int min_pts);
    
    static std::vector<Cluster> cluster(const std::vector<double>& errors, 
                                       double eps = 5.0, 
                                       int min_points = 3);
    
    std::vector<int> fit_predict(const std::vector<double>& data);
    
private:
    double eps_;
    int min_pts_;
    int next_cluster_id_;
    
    std::vector<int> region_query(const std::vector<double>& data, size_t point_idx) const;
    void expand_cluster(const std::vector<double>& data, std::vector<int>& labels, 
                       size_t point_idx, int cluster_id);
};