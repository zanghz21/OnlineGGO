#pragma once

#include "common.h"
#include "nlohmann/json.hpp"

// using json = nlohmann::json;

class Grid
{
public:
    Grid(string fname,
         double left_w_weight = 1,
         double right_w_weight = 1);
    Grid(nlohmann::json map_json,
         double left_w_weight = 1,
         double right_w_weight = 1);

    Grid() {}

    void load_map_from_path(string fname,
                            double left_w_weight = 1,
                            double right_w_weight = 1);
    void load_map_from_json(nlohmann::json map_json,
                            double left_w_weight = 1,
                            double right_w_weight = 1);

    int rows = 0;
    int cols = 0;
    std::vector<int> map;
    string map_name;

    std::vector<int> end_points;
    std::vector<int> agent_home_locations;
    std::vector<int> empty_locations;

    std::vector<char> grid_types;

    std::vector<double> agent_home_loc_weights;
};
