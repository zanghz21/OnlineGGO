#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include "CompetitionSystem.h"
#include "nlohmann/json.hpp"
#include "dev.h"
#include <signal.h>
#include <climits>
#include <memory>
#include <random>
#include <ctime>
#include <chrono>

namespace po = boost::program_options;
using json = nlohmann::json;
std::unique_ptr<BaseSystem> py_system_ptr;

// cpp simulator for on+GPIBT and off+GPIBT 
void gen_random_instance(Grid & grid, std::vector<int> & agents, std::vector<int> & tasks, 
    int num_agents, int num_tasks, uint seed) {
    std::mt19937 MT(seed);

    std::vector<int> empty_locs;
    for (int i=0;i<grid.map.size();++i) {
        if (grid.map[i] == 0) {
            empty_locs.push_back(i);
        }
    }

    std::shuffle(empty_locs.begin(), empty_locs.end(), MT);

    for (int i=0;i<num_agents;++i) {
        agents.push_back(empty_locs[i]);
    }

    if (grid.end_points.size()>0 && grid.sortation_points.size()>0) {
        // only sample goal locations from end_points
        std::cout<<"sample goal locations from end points"<<std::endl;
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i=0;i<num_tasks;++i) {
            double p = dis(MT);
            if (p < 0.5){
                int rnd_idx=MT()%grid.end_points.size();
                tasks.push_back(grid.end_points[rnd_idx]);
            } else {
                int rnd_idx=MT()%grid.sortation_points.size();
                tasks.push_back(grid.sortation_points[rnd_idx]);
            }
        }        
    } else {
        std::cout<<"sample goal locations from empty locations"<<std::endl;
        for (int i=0;i<num_tasks;++i) {
            int rnd_idx=MT()%empty_locs.size();
            tasks.push_back(empty_locs[rnd_idx]);
        }
    }
}

// TODO: SHOULD HAVE INTERFACE TO TRANSFER NET WEIGHTS INTO NETWORK!!!
std::string run(const pybind11::kwargs& kwargs)
{
    int plan_time_limit = 10;
    Logger* logger = new Logger();
    MAPFPlanner* planner = nullptr;
    planner = new MAPFPlanner();

    std::string inputFilePath = "";

    bool gen_tasks = false;
    if (kwargs.contains("gen_tasks")){
        gen_tasks = kwargs["gen_tasks"].cast<bool>();
    }
    Grid grid;
    std::string map_path;
    std::vector<int> agents;
    std::vector<int> tasks;
    std::string task_assignment_strategy;
    int num_tasks_reveal = 1;
    if (!gen_tasks){
        if (kwargs.contains("all_json_path")){
            inputFilePath = kwargs["all_json_path"].cast<std::string>();
        } else {
            std::cout << "kwargs must contain [all_json_path]! " << std::endl;
            exit(-1);
        }
        std::cout << "map path =" << inputFilePath << std::endl;
        boost::filesystem::path p(inputFilePath);
        boost::filesystem::path dir = p.parent_path();
        std::string base_folder = dir.string();
        if (base_folder.size() > 0 && base_folder.back()!='/'){
            base_folder += "/";
        }
        std::cout << base_folder << std::endl;

        json data;
        std::ifstream f(inputFilePath);
        try{
            data = json::parse(f);
        }
        catch(json::parse_error error ) {
            std::cerr << "Failed to load " << inputFilePath << std::endl;
            std::cerr << "Message: " << error.what() << std::endl;
            exit(1);
        }
        map_path = read_param_json<std::string>(data, "mapFile");
        grid.load_from_path(base_folder + map_path);

        int team_size = read_param_json<int>(data, "teamSize");
        agents = read_int_vec(base_folder + read_param_json<std::string>(data, "agentFile"),team_size);
        tasks = read_int_vec(base_folder + read_param_json<std::string>(data, "taskFile"));
        task_assignment_strategy = data["taskAssignmentStrategy"].get<std::string>();

        num_tasks_reveal = read_param_json<int>(data, "numTasksReveal", 1);
    } else {
        if (!kwargs.contains("map_path")){
            std::cout << "kwargs must contain [map_path]! " << std::endl;
            exit(-1);
        }
        map_path = kwargs["map_path"].cast<std::string>();
        int num_agents, num_tasks;
        uint seed;
        if (kwargs.contains("num_agents")){
            num_agents = kwargs["num_agents"].cast<int>();
        } else {
            std::cout << "kwargs must contain [num_agents]! " << std::endl;
            exit(-1);
        }
        if (kwargs.contains("num_tasks")){
            num_tasks = kwargs["num_tasks"].cast<int>();
        } else {
            std::cout << "kwargs must contain [num_tasks]! " << std::endl;
            exit(-1);
        }
        if (kwargs.contains("seed")){
            seed = kwargs["seed"].cast<uint>();
        } else {
            std::cout << "kwargs must contain [seed]! " << std::endl;
            exit(-1);
        }
        if (kwargs.contains("task_assignment_strategy")){
            task_assignment_strategy = kwargs["task_assignment_strategy"].cast<std::string>();
        } else {
            std::cout << "kwargs must contain [task_assignment_strategy]! " << std::endl;
            exit(-1);
        }
        if (kwargs.contains("num_tasks_reveal")){
            num_tasks_reveal = kwargs["num_tasks_reveal"].cast<int>();
        }
        grid.load_from_path(map_path);
        // std::cout << grid.agent_home_locs.size() <<", "<< grid.sortation_points.size() <<", "<< grid.end_points.size()<<std::endl;
        if (grid.agent_home_locs.size() == 0){
            gen_random_instance(grid, agents, tasks, num_agents, num_tasks, seed);
        }
    }
    
    // TODO: should rewrite here!!!
    planner->env->map_name = map_path.substr(map_path.find_last_of("/") + 1);
    planner->env->file_storage_path = "";
    
    if (kwargs.contains("past_traffic_interval")){
        int past_traffic_interval = kwargs["past_traffic_interval"].cast<int>();
        planner->env->past_traffic_interval = past_traffic_interval;
    }

    // set net params
    if (kwargs.contains("net_input_type")){
        std::string input_type = kwargs["net_input_type"].cast<std::string>();
        network_config.input_type = input_type;
        if (input_type == "traf_and_goal" || input_type == "flow_and_traf_and_goal"){
            if (planner->env->past_traffic_interval < 0){
                std::cout << "if use past traffic as input, past traffic interval should > 0!"<<std::endl;
            }
        }
    }
    if (kwargs.contains("use_cached_nn")){
        bool use_cached_nn = kwargs["use_cached_nn"].cast<bool>();
        network_config.use_cached_nn = use_cached_nn;
    }
    if (kwargs.contains("use_all_flow")){
        bool use_all_flow = kwargs["use_all_flow"].cast<bool>();
        network_config.use_all_flow = use_all_flow;
    }
    if (kwargs.contains("has_map")){
        bool has_map = kwargs["has_map"].cast<bool>();
        network_config.has_map = has_map;
        // std::cout << "has_map =" << has_map <<std::endl;
    }
    if (kwargs.contains("has_path")){
        bool has_path = kwargs["has_path"].cast<bool>();
        network_config.has_path = has_path;
    }
    if (kwargs.contains("has_previous")){
        bool has_prev = kwargs["has_previous"].cast<bool>();
        network_config.has_previous = has_prev;
    }
    if (kwargs.contains("output_size")){
        int output_size = kwargs["output_size"].cast<int>();
        network_config.output_size = output_size;
    }
    if (kwargs.contains("hidden_size")){
        int hidden_size = kwargs["hidden_size"].cast<int>();
        network_config.hidden_size = hidden_size;
    }
    if (kwargs.contains("win_r")){
        int win_r = kwargs["win_r"].cast<int>();
        network_config.WIN_R = win_r;
    }
    if (kwargs.contains("rotate_input")){
        bool rotate_input = kwargs["rotate_input"].cast<bool>();
        network_config.use_rotate_input = rotate_input;
    }
    if (kwargs.contains("default_obst_flow")){
        double default_obst_flow = kwargs["default_obst_flow"].cast<double>();
        network_config.default_obst_flow = default_obst_flow;
    }
    if (kwargs.contains("learn_obst_flow")){
        bool learn_obst_flow = kwargs["learn_obst_flow"].cast<bool>();
        network_config.learn_obst_flow = learn_obst_flow;
    }
    if (kwargs.contains("net_type")){
        std::string net_type = kwargs["net_type"].cast<std::string>();
        planner->set_network_type(net_type);
    }
    if (kwargs.contains("network_params")){
        // std::cout << "has nn params" <<std::endl;
        std::string netparams_str=kwargs["network_params"].cast<std::string>();
        nlohmann::json netparams_json=nlohmann::json::parse(netparams_str);
        std::vector<double> network_params;
        for (auto & p:netparams_json) {
            network_params.push_back(p.get<double>());
        }
        ONLYDEV(std::cout<<network_params.size()<<std::endl;
        for (unsigned int i=0; i<network_params.size(); ++i){
            std::cout<<network_params[i]<<", ";
        }
        std::cout<<std::endl;)
        // std::cout << "here"<<std::endl;
        planner->set_network_params(network_params);
        // exit(1);
    }

    // set dist params
    if (kwargs.contains("dist_sigma")){
        dist_params.sigma = kwargs["dist_sigma"].cast<double>();
    }
    if (kwargs.contains("dist_K")){
        dist_params.K = kwargs["dist_K"].cast<int>();
    }

    if (kwargs.contains("map_weights")){
        std::string weight_str=kwargs["map_weights"].cast<std::string>();
        nlohmann::json weight_json=nlohmann::json::parse(weight_str);
        std::vector<double> map_weights;
        for (auto & w:weight_json) {
            map_weights.push_back(w.get<float>());
        }
        planner->env->map_weights = map_weights;
    }

    ActionModelWithRotate* model = new ActionModelWithRotate(grid);
    // model->set_logger(logger);

    if (grid.agent_home_locs.size() == 0){
        if (task_assignment_strategy=="greedy"){
            py_system_ptr = std::make_unique<TaskAssignSystem>(grid, planner, agents, tasks, model);
        } else if (task_assignment_strategy=="roundrobin"){
            py_system_ptr = std::make_unique<InfAssignSystem>(grid, planner, agents, tasks, model);
        } else if (task_assignment_strategy=="roundrobin_fixed"){
            std::vector<vector<int>> assigned_tasks(agents.size());
            for(int i = 0; i < tasks.size(); i++){
                assigned_tasks[i%agents.size()].push_back(tasks[i]);
            }
            py_system_ptr = std::make_unique<FixedAssignSystem>(grid, planner, agents, assigned_tasks, model);
        } else if (task_assignment_strategy == "online_generate"){
            uint seed=kwargs["seed"].cast<uint>();
            py_system_ptr = std::make_unique<OnlineGenerateTaskSystem>(grid, planner, agents, model, seed);
        } else{
            std::cerr << "unknown task assignment strategy " << task_assignment_strategy << std::endl;
            logger->log_fatal("unknown task assignment strategy " + task_assignment_strategy);
            exit(1);
        }
    } else {
        int num_agents=kwargs["num_agents"].cast<int>();
        std::cout << "using kiva system (random task generation) with "<<num_agents<<" agents"<<std::endl;
        
        uint seed=kwargs["seed"].cast<uint>();
        py_system_ptr = std::make_unique<KivaSystem>(grid,planner,model,num_agents,seed);
    }

    if(kwargs.contains("task_dist_change_interval")){
        int interval = kwargs["task_dist_change_interval"].cast<int>();
        py_system_ptr->task_dist_change_interval = interval;
    }

    if(kwargs.contains("task_random_type")){
        std::string random_type = kwargs["task_random_type"].cast<std::string>();
        py_system_ptr->set_random_type(random_type);
    }

    // py_system_ptr->set_logger(logger);
    py_system_ptr->set_plan_time_limit(plan_time_limit);
    py_system_ptr->set_preprocess_time_limit(INT_MAX);

    py_system_ptr->set_num_tasks_reveal(num_tasks_reveal);

    // signal(SIGINT, sigint_handler);
    int simuTime = 1000;
    if (kwargs.contains("simu_time")){
        simuTime = kwargs["simu_time"].cast<int>();
    }

    if (simuTime == -1){
        simuTime = (grid.cols + grid.rows)/2 * 10;
    }
    py_system_ptr->simulation_time = simuTime;

    std::cout << "start simulation!" <<std::endl;
    // clock_t start_time = clock();
    auto start_time = std::chrono::high_resolution_clock::now();
    py_system_ptr->simulate(simuTime);
    // double runtime = (double)(clock() - start_time)/ CLOCKS_PER_SEC;
    auto end_time = std::chrono::high_resolution_clock::now();
    // py_system_ptr->saveResults("./exp/test.json");
    std::chrono::duration<double> runtime = end_time - start_time;
    nlohmann::json result_json = py_system_ptr->analyzeResults(simuTime);
    result_json["cpu_runtime"] = runtime.count();
    
    bool save_result = false;
    std::string save_path = "";
    if (kwargs.contains("save_path")){
        save_result = true;
        save_path = kwargs["save_path"].cast<std::string>();
    }
    
    if (save_result){
        py_system_ptr->saveResults(save_path);
    }
    return result_json.dump(4);
}

string playground(){
	std::string json_string = R"(
	{
		"pi": 3.141,
		"happy": true
	}
	)";
	json ex1 = json::parse(json_string);

	std::cout << ex1["pi"] << std::endl;

	return ex1.dump();
}

PYBIND11_MODULE(py_driver, m){
    m.def("playground", &playground, "Playground function to test everything");
    m.def("run", &run, "Function to run warehouse simulation");
}