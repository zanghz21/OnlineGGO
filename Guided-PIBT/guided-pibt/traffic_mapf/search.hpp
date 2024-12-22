
#ifndef search_hpp
#define search_hpp

#include "Types.h"
#include "utils.hpp"
#include "Memory.h"
#include "heap.h"
#include "search_node.h"
#include "heuristics.hpp"
#include "network.hpp"
#include "flow.hpp"
#include <unordered_map>
#include <chrono>

namespace TrafficMAPF{
//a astar minimized the opposide traffic flow with existing traffic flow

s_node singleShortestPath(SharedEnvironment* env, std::vector<Int4>& flow,
    HeuristicTable& ht,std::vector<int>& traffic, Traj& traj,
    MemoryPool& mem, int start, int goal);

s_node aStarOF(SharedEnvironment* env, std::vector<Int4>& flow, int flow_sum, 
    HeuristicTable& ht,std::vector<int>& traffic, Traj& traj,
    MemoryPool& mem, int start, int goal, std::shared_ptr<Network> network_ptr, 
    std::vector<std::pair<bool, std::vector<double>>>& nn_cached_value);

void get_flow_NN_input(SharedEnvironment* env, std::vector<Int4>& flow, int all_flow, 
    int center_id, std::vector<double>& flow_input, std::vector<double>& map_input, std::vector<int>& ids);

void get_traf_and_goal_NN_input(SharedEnvironment* env, int center_id, 
    std::vector<double>& nn_input, std::vector<double>& map_input, std::vector<int>& ids);

void get_flow_and_traf_and_goal_NN_input(SharedEnvironment* env, std::vector<Int4>& flow, int all_flow, 
    int center_id, std::vector<double>& flow_input, std::vector<double>& map_input, std::vector<int>& ids);

void get_minimum_NN_input(SharedEnvironment* env, std::vector<Int4>& flow, int all_flow, 
    int center_id, std::vector<double>& flow_input, std::vector<double>& map_input, std::vector<int>& ids);
}
#endif