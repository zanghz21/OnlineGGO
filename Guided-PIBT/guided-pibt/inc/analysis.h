#pragma once
#include "nlohmann/json.hpp"
#include "Grid.h"
#include <vector>
#include <iostream>

nlohmann::json analyze_extra_json(const nlohmann::json & result, const Grid & grid);
