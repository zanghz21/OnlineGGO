#pragma once

#include "common.h"

class Grid{
public:
    Grid(){};
    Grid(string fname);

    void load_from_path(string filename);

    int rows = 0;
    int cols = 0;
    std::vector<int> map;
    string map_name;
    std::vector<int> empty_locs;
    std::vector<int> end_points;
    std::vector<int> sortation_points;
    std::vector<int> agent_home_locs;
    std::vector<double> agent_home_loc_weights;
    std::vector<char> grid_types;
};
