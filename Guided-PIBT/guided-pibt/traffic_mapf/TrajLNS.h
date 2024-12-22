#ifndef TRAJ_LNS_H
#define TRAJ_LNS_H

#include "Types.h"
#include "Memory.h"
#include "search_node.h"
#include "heap.h"
#include "network.hpp"

#include <set>

namespace TrafficMAPF{
enum ADAPTIVE {RANDOM, CONGESTION, COUNT};

struct FlowHeuristic{
    HeuristicTable* h; 
    int target;
    int origin;
    pqueue_min_of open;
    MemoryPool mem;


    bool empty(){
        return mem.generated() == 0;
    }
    void reset(){
        // op_flows.clear();
        // depths.clear();
        // dists.clear();
        open.clear();
        mem.reset();
    }

};

class TrajLNS{
    public:
    SharedEnvironment* env;
    std::vector<int> tasks;

    TimePoint start_time;
    int t_ms=0;


    std::vector<Traj> trajs;
    std::vector<Int4> flow;
    int flow_sum;
    
    // int flow_thres_stride;
    // std::pair<int, int> flow_sum_thres;
    std::vector<std::pair<bool, std::vector<double>>> nn_cached_value; // (valid, nn_cost), if invalid, then need recompute nn_cost

    std::vector<HeuristicTable> heuristics;
    std::vector<Dist2Path> traj_dists;
    std::vector<s_node> goal_nodes;// store the goal node of single agent search for each agent. contains all cost information.
    // DH dh;
    std::vector<FlowHeuristic> flow_heuristics;

    std::vector<double> weights; //weights for adaptive lns

    double decay_factor = 0.001;
    double reaction_factor = 0.1;

    int group_size = LNS_GROUP_SIZE;


    std::vector<std::set<int>> occupations;
    std::vector<bool> tabu_list;
    int num_in_tablu=0;

    int traj_inited = 0;
    int dist2path_inited = 0;
    int tdh_build = 0;

    int op_flow = 0;
    int vertex_flow = 0;
    double nn_flow = 0.0; 
    int soc = 0;

    MemoryPool mem;

    void init_mem(){
        mem.init(env->map.size());
    }

    TrajLNS(SharedEnvironment* env):
        env(env),
        trajs(env->num_of_agents), tasks(env->num_of_agents),tabu_list(env->num_of_agents,false),
        flow(env->map.size(),Int4({0,0,0,0})), heuristics(env->map.size()),
        flow_sum(0),  
        nn_cached_value(env->map.size(), std::make_pair(false, std::vector<double>(4, 0.0))), 
        flow_heuristics(env->num_of_agents),
        traj_dists(env->num_of_agents),goal_nodes(env->num_of_agents),occupations(env->map.size()){
            weights.resize(ADAPTIVE::COUNT,1.0);
            
            // (cols + rows)/2 should be substitudes with avg ES dist, if rigorous; 5 for time steps
            // this->flow_thres_stride = std::min(RELAX, this->env->num_of_agents) 
            //                         * (this->env->cols + this->env->rows)/2 * 5;
            // this->flow_sum_thres = std::make_pair(0, this->flow_thres_stride);
        };

    TrajLNS(){};

    void set_nn_cached_invalid(int id){
        // std::cout << "curr thres: "<<this->flow_sum<<", "<<this->flow_sum_thres.first<<", "<<this->flow_sum_thres.second<<std::endl;
        // if (this->flow_sum < this->flow_sum_thres.first){
        //     this->flow_sum_thres.first -= this->flow_thres_stride/2;
        //     this->flow_sum_thres.second -= this->flow_thres_stride/2;
        //     this->nn_cached_value = std::vector<std::pair<bool, std::vector<double>>>(
        //         this->env->map.size(), std::make_pair(false, std::vector<double>(4, 0.0))
        //     );
        //     std::cout << "update thres: "<<this->flow_sum<<", "<<this->flow_sum_thres.first<<", "<<this->flow_sum_thres.second<<std::endl;
        //     return;
        // }
        // if (this->flow_sum > this->flow_sum_thres.second){
        //     this->flow_sum_thres.first += this->flow_thres_stride/2;
        //     this->flow_sum_thres.second += this->flow_thres_stride/2;
        //     this->nn_cached_value = std::vector<std::pair<bool, std::vector<double>>>(
        //         this->env->map.size(), std::make_pair(false, std::vector<double>(4, 0.0))
        //     );
        //     std::cout << "update thres: "<<this->flow_sum<<", "<<this->flow_sum_thres.first<<", "<<this->flow_sum_thres.second<<std::endl;
        //     return;
        // }

        int WIN_R = network_config.WIN_R;
        for (int k1=-WIN_R; k1<=WIN_R; ++k1){
            for (int k2=-WIN_R; k2<=WIN_R; ++k2){
                int temp_id = id + k1*env->cols + k2;
                if (temp_id <0 || temp_id>=this->env->map.size()){
                    continue;
                }
                this->nn_cached_value[temp_id].first = false;
            }
        }
    };

    void reset_nn_cache(){
        nn_cached_value.clear();
        nn_cached_value.resize(env->map.size(), std::make_pair(false, std::vector<double>(4, 0.0)));
    }
};
}
#endif