#pragma once

#include "Grid.h"
#include <vector>
#include <cmath>
#include <random>

struct DistributionParams{
    double sigma = 0.5;
    int K = 3;
};
extern DistributionParams dist_params;

void generateTaskAndAgentGaussianDist(const Grid& map, std::mt19937& MT, vector<double>& w_dist, vector<double>& e_dist);
void generateTaskAndAgentMultiModeGaussianDist(const Grid& map, std::mt19937& MT, vector<double>& w_dist, vector<double>& e_dist, int K=dist_params.K);
void generateTaskAndAgentGaussianEmptyDist(const Grid& map, std::mt19937& MT, vector<double>& empty_weights);
void generateTaskAndAgentLRDist(const Grid& map, std::mt19937& MT, vector<double>& w_dist, vector<double>& e_dist);
void generateMultiModeGaussianEmptyDist(const Grid& map, std::mt19937& MT,  vector<double>& empty_weights, int K=dist_params.K);