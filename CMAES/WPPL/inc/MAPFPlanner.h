#pragma once

#ifndef NO_ROT

#include <ctime>
#include "SharedEnv.h"
#include "ActionModel.h"
#include "RHCR/interface/RHCRSolver.h"
#include "nlohmann/json.hpp"
#include "RHCR/main/SingleAgentSolver.h"
#include "RHCR/main/MAPFSolver.h"
#include "RHCR/main/WHCAStar.h"
#include "RHCR/main/ECBS.h"
#include "RHCR/main/LRAStar.h"
#include "RHCR/main/PBS.h"
#include "RHCR/main/ID.h"
#include "RHCR/interface/RHCRSolver.h"
#include "RHCR/interface/CompetitionGraph.h"
#include "common.h"
#include <memory>
#include "LaCAM2/LaCAM2Solver.hpp"
#include "LNS/LNSSolver.h"
// #include "util/network.h"

class MAPFPlanner
{
public:

    SharedEnvironment* env;

	MAPFPlanner(SharedEnvironment* env): env(env){};
    MAPFPlanner(){env = new SharedEnvironment();};
	virtual ~MAPFPlanner(){delete env;};


    virtual void initialize(int preprocess_time_limit);

    // return next states for all agents
    // virtual void plan(int time_limit, std::vector<Action> & plan);
    virtual void plan(int time_limit,vector<Action> & actions, vector<list<State>>& cur_exec_paths, vector<list<State>>& cur_plan_paths);


    // Start kit dummy implementation
    std::list<pair<int,int>>single_agent_plan(int start,int start_direct, int end);
    int getManhattanDistance(int loc1, int loc2);
    std::list<pair<int,int>> getNeighbors(int location, int direction);
    bool validateMove(int loc,int loc2);

    // void init_network(int map_h, int map_w){
    //     this->cnn_net = std::shared_ptr<CNNNetwork>(new CNNNetwork(map_h, map_w));
    // }
    // void set_network_params(std::vector<double>& network_params);

    bool consider_rotation=true;
    string lifelong_solver_name;
    std::shared_ptr<RHCR::RHCRSolver> rhcr_solver;
    std::shared_ptr<LaCAM2::LaCAM2Solver> lacam2_solver;
    std::shared_ptr<LNS::LNSSolver> lns_solver;

    double max_step_time=0;

    std::shared_ptr<std::vector<float> > map_weights;
    nlohmann::json config;
    void load_configs();
    std::string load_map_weights(string weights_path);

    RHCR::MAPFSolver* rhcr_build_mapf_solver(nlohmann::json & config, RHCR::CompetitionGraph & graph);
    void rhcr_config_solver(std::shared_ptr<RHCR::RHCRSolver> & solver,nlohmann::json & config);

    int max_execution_steps;
    // std::shared_ptr<CNNNetwork> cnn_net = std::make_shared<CNNNetwork>();
};

#else

#include <ctime>
#include "SharedEnv.h"
#include "ActionModel.h"
#include "nlohmann/json.hpp"
#include "common.h"
#include <memory>
#include "LaCAM2/LaCAM2Solver.hpp"
#include "LNS/LNSSolver.h"
class MAPFPlanner
{
public:

    SharedEnvironment* env;

	MAPFPlanner(SharedEnvironment* env): env(env){};
    MAPFPlanner(){env = new SharedEnvironment();};
	virtual ~MAPFPlanner(){delete env;};


    virtual void initialize(int preprocess_time_limit);
    virtual void update();

    // return next states for all agents
    // virtual void plan(int time_limit, std::vector<Action> & plan);
    virtual void plan(int time_limit,vector<Action> & actions, vector<list<State>>& cur_exec_paths, vector<list<State>>& cur_plan_paths);

    string lifelong_solver_name;
    std::shared_ptr<LaCAM2::LaCAM2Solver> lacam2_solver;
    std::shared_ptr<LNS::LNSSolver> lns_solver;

    bool h_update_late=true;

    std::shared_ptr<std::vector<float> > map_weights;
    nlohmann::json config;
    void load_configs();
    std::string load_map_weights(string weights_path);
    
    int max_execution_steps;
    // void set_network_params(std::vector<double>& network_params);
    // void init_network(int map_h, int map_w){
    //     this->cnn_net = std::shared_ptr<CNNNetwork>(new CNNNetwork(map_h, map_w));
    // }
    // std::shared_ptr<CNNNetwork> cnn_net = std::make_shared<CNNNetwork>();
};

#endif