#pragma once

#include "Grid.h"
#include <vector>
#include <cmath>
#include <random>

struct DistributionParams{
    double sigma = 0.5;
    int K = 3;
    double k_ratio_low = 0.0;
    double k_ratio_high = 0.1;
};
extern DistributionParams dist_params;

void generateTaskAndAgentGaussianDist(const Grid& map, std::mt19937& MT, vector<double>& w_dist, vector<double>& e_dist);
void generateTaskAndAgentMultiModeGaussianDist(const Grid& map, std::mt19937& MT, vector<double>& w_dist, vector<double>& e_dist, int K=dist_params.K);
void generateTaskAndAgentMultiModeGaussianRandomKDist(const Grid& map, std::mt19937& MT, vector<double>& w_dist, vector<double>& e_dist);
void generateTaskAndAgentGaussianEmptyDist(const Grid& map, std::mt19937& MT, vector<double>& empty_weights);
void generateMultiModeGaussianEmptyDist(const Grid& map, std::mt19937& MT,  vector<double>& empty_weights, int K=dist_params.K);
void generateTaskAndAgentLRDist(const Grid& map, std::mt19937& MT, vector<double>& w_dist, vector<double>& e_dist);
