#pragma once
#include "common.h"
#include "nlohmann/json.hpp"
#include "Grid.h"

int get_orient_idx(string o);
void move(int & x,int & y,int & o, char action);
nlohmann::json analyze_result_json(const nlohmann::json & result, Grid & grid);
nlohmann::json analyze_curr_result_json(const nlohmann::json & result, Grid & grid);