#include "CompetitionSystem.h"
#include "Evaluation.h"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include "nlohmann/json.hpp"
#include <signal.h>
#include <ctime>
#include <climits>
#include <memory>
#include <util/Analyzer.h>
#include "nlohmann/json.hpp"

#ifdef MAP_OPT
#include "util/analyze.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

namespace po = boost::program_options;
using json = nlohmann::json;

namespace py = pybind11;

// po::variables_map vm;
std::unique_ptr<BaseSystem> system_ptr;


// void sigint_handler(int a)
// {
//     fprintf(stdout, "stop the simulation...\n");
//     if (!vm["evaluationMode"].as<bool>())
//     {
//         system_ptr->saveResults(vm["output"].as<std::string>());
//     }
//     _exit(0);
// }


int _get_Manhattan_distance(int loc1, int loc2, int cols) {
    return abs(loc1 / cols - loc2 / cols) + abs(loc1 % cols - loc2 % cols);
}

std::shared_ptr<std::vector<float> > weight_format_conversion(Grid & grid, std::vector<float> & weights)
{
    const int max_weight=100000;
    std::shared_ptr<std::vector<float> > map_weights_ptr = std::make_shared<std::vector<float> >(grid.map.size()*5, max_weight);
    auto & map_weights=*map_weights_ptr;

    const int dirs[4]={1,-grid.cols, -1, grid.cols};
    const int map_weights_idxs[4]={0,3,2,1};

    int j=0;

    ++j; // the 0 indexed weight is for wait

    for (int i=0;i<grid.map.size();++i) {
        if (grid.map[i] == 1) {
            continue;
        }

        map_weights[i*5+4] = weights[0];

        for (int d=0;d<4;++d) {
            int dir=dirs[d];
            if (
                0<=i+dir && i+dir<grid.map.size() &&
                _get_Manhattan_distance(i, i+dir, grid.cols) <= 1 &&
                grid.map[i+dir] != 1
            ) {
                float weight = weights.at(j);
                if (weight==-1) {
                    weight=max_weight;
                }

                int map_weight_idx=map_weights_idxs[d];
                map_weights[i*5+map_weight_idx]=weight;
                ++j;
            }
        }
    }

    // std::cout<<"map weights: ";
    // for (auto i=0;i<map_weights.size();++i) {
    //     std::cout<<map_weights[i]<<" ";
    // }
    // std::cout<<endl;

    if (j!=weights.size()) {
        std::cout<<"weight size mismatch: "<<j<<" vs "<<weights.size()<<std::endl;
        exit(1);
    }

    return map_weights_ptr;
}


std::shared_ptr<std::vector<float> > weight_format_conversion_with_wait_costs(Grid & grid, std::vector<float> & edge_weights, std::vector<float> & wait_costs)
{
    const int max_weight=100000;
    std::shared_ptr<std::vector<float> > map_weights_ptr = std::make_shared<std::vector<float> >(grid.map.size()*5, max_weight);
    auto & map_weights=*map_weights_ptr;

    const int dirs[4]={1,-grid.cols, -1, grid.cols};
    const int map_weights_idxs[4]={0,3,2,1};

    // read wait cost
    int j=0;
    for (int i=0;i<grid.map.size();++i) {
        if (grid.map[i] == 1) {
            continue;
        }
        map_weights[i*5+4] = wait_costs[j];
        ++j;
    }

    if (j!=wait_costs.size()) {
        std::cout<<"wait cost size mismatch: "<<j<<" vs "<<wait_costs.size()<<std::endl;
        exit(1);
    }


    // read edge cost
    j=0;
    for (int i=0;i<grid.map.size();++i) {
        if (grid.map[i] == 1) {
            continue;
        }

        for (int d=0;d<4;++d) {
            int dir=dirs[d];
            if (
                0<=i+dir && i+dir<grid.map.size() &&
                _get_Manhattan_distance(i, i+dir, grid.cols) <= 1 &&
                grid.map[i+dir] != 1
            ) {
                float weight = edge_weights.at(j);
                if (weight==-1) {
                    weight=max_weight;
                }

                int map_weight_idx=map_weights_idxs[d];
                map_weights[i*5+map_weight_idx]=weight;
                ++j;
            }
        }
    }

    // std::cout<<"map weights: ";
    // for (auto i=0;i<map_weights.size();++i) {
    //     std::cout<<map_weights[i]<<" ";
    // }
    // std::cout<<endl;

    if (j!=edge_weights.size()) {
        std::cout<<"edge weight size mismatch: "<<j<<" vs "<<edge_weights.size()<<std::endl;
        exit(1);
    }

    return map_weights_ptr;
}


void gen_random_instance(Grid & grid, std::vector<int> & agents, std::vector<int> & tasks, int num_agents, int num_tasks, uint seed) {
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

    if (grid.end_points.size()>0) {
        // only sample goal locations from end_points
        std::cout<<"sample goal locations from end points"<<std::endl;
        for (int i=0;i<num_tasks;++i) {
            int rnd_idx=MT()%grid.end_points.size();
            tasks.push_back(grid.end_points[rnd_idx]);
        }        
    } else {
        std::cout<<"sample goal locations from empty locations"<<std::endl;
        for (int i=0;i<num_tasks;++i) {
            int rnd_idx=MT()%empty_locs.size();
            tasks.push_back(empty_locs[rnd_idx]);
        }
    }
}

std::string run(const py::kwargs& kwargs)
{    
    int simulation_steps=kwargs["simulation_steps"].cast<int>();
    double plan_time_limit=kwargs["plan_time_limit"].cast<double>();
    double preprocess_time_limit=kwargs["preprocess_time_limit"].cast<double>();
    // std::string map_path=kwargs["map_path"].cast<std::string>();
    std::string file_storage_path=kwargs["file_storage_path"].cast<std::string>();
    std::string task_assignment_strategy=kwargs["task_assignment_strategy"].cast<std::string>();
    int num_tasks_reveal=kwargs["num_tasks_reveal"].cast<int>();

    // Read in left and right weights
    double left_w_weight = 1;
    double right_w_weight = 1;
    if (kwargs.contains("left_w_weight"))
    {
        left_w_weight = kwargs["left_w_weight"].cast<double>();
    }
    if (kwargs.contains("right_w_weight"))
    {
        right_w_weight = kwargs["right_w_weight"].cast<double>();
    }

    // Read in map. Use map_path by default. If not provided, use map_json
    Grid grid;
    if (kwargs.contains("map_path"))
    {
        std::string map_path = kwargs["map_path"].cast<std::string>();
        grid.load_map_from_path(map_path, left_w_weight, right_w_weight);
    }
    else if (kwargs.contains("map_json_str") ||
             kwargs.contains("map_json_path"))
    {
        json map_json;
        if (kwargs.contains("map_json_str"))
        {
            std::string map_json_str = kwargs["map_json_str"].cast<std::string>();
            map_json = json::parse(map_json_str);
        }
        else
        {
            std::string map_json_path = kwargs["map_json_path"].cast<std::string>();
            std::ifstream f(map_json_path);
            map_json = json::parse(f);
        }
        grid.load_map_from_json(map_json, left_w_weight, right_w_weight);
    }
    

    Logger *logger = new Logger();
    // if (vm.count("logFile"))
    //     logger->set_logfile(vm["logFile"].as<std::string>());

    MAPFPlanner *planner = nullptr;
    // Planner is inited here, but will be managed and deleted by system_ptr deconstructor
    // if (vm["evaluationMode"].as<bool>())
    // {
    //     logger->log_info("running the evaluation mode");
    //     planner = new DummyPlanner(vm["output"].as<std::string>());
    // }
    // else
    // {
        planner = new MAPFPlanner();
    // }

    // auto input_json_file = vm["inputFile"].as<std::string>();
    // json data;
    // std::ifstream f(input_json_file);
    // try
    // {
    //     data = json::parse(f);
    // }
    // catch (json::parse_error error)
    // {
    //     std::cerr << "Failed to load " << input_json_file << std::endl;
    //     std::cerr << "Message: " << error.what() << std::endl;
    //     exit(1);
    // }

    // auto map_path = read_param_json<std::string>(data, "mapFile");
    // Grid grid(base_folder + map_path);
    // Grid grid(map_path);

    // planner->env->map_name = map_path.substr(map_path.find_last_of("/") + 1);
    planner->env->map_name = grid.map_name;
    // planner->env->file_storage_path = vm["fileStoragePath"].as<std::string>();
    planner->env->file_storage_path = file_storage_path;

    if (kwargs.contains("h_update_late")){
        bool h_update_late = kwargs["h_update_late"].cast<bool>();
        planner->h_update_late = h_update_late;
    }

    if (kwargs.contains("config")){
        std::string config_str=kwargs["config"].cast<std::string>();
        nlohmann::json config=nlohmann::json::parse(config_str);
        planner->config=config;
    }

    if (kwargs.contains("weights")) {
        std::string weight_str=kwargs["weights"].cast<std::string>();
        nlohmann::json weight_json=nlohmann::json::parse(weight_str);
        std::vector<float> weights;
        for (auto & w:weight_json) {
            weights.push_back(w.get<float>());
        }

        if (kwargs.contains("wait_costs")) {
            std::string wait_costs_str=kwargs["wait_costs"].cast<std::string>();
            nlohmann::json wait_costs_json=nlohmann::json::parse(wait_costs_str);
            std::vector<float> wait_costs;
            for (auto & w:wait_costs_json) {
                wait_costs.push_back(w.get<float>());
            }


            planner->map_weights=weight_format_conversion_with_wait_costs(grid, weights, wait_costs);
        } else {
            planner->map_weights=weight_format_conversion(grid, weights);
        }

    } 

    // if (kwargs.contains("network_params")){
    //     // std::cout << "has nn params" <<std::endl;
    //     std::string netparams_str=kwargs["network_params"].cast<std::string>();
    //     nlohmann::json netparams_json=nlohmann::json::parse(netparams_str);
    //     std::vector<double> network_params;
    //     for (auto & p:netparams_json) {
    //         network_params.push_back(p.get<double>());
    //     }
    //     ONLYDEV(std::cout<<network_params.size()<<std::endl;
    //     for (unsigned int i=0; i<network_params.size(); ++i){
    //         std::cout<<network_params[i]<<", ";
    //     }
    //     std::cout<<std::endl;)
    //     // std::cout << "here"<<std::endl;
    //     planner->init_network(grid.rows, grid.cols);
    //     planner->set_network_params(network_params);
    //     exit(1);
    // }
    
    ActionModelWithRotate *model = new ActionModelWithRotate(grid);
    model->set_logger(logger);

    if (grid.agent_home_locations.size()==0 || grid.end_points.size()==0) {
        std::vector<int> agents;
        std::vector<int> tasks;

        if (!kwargs.contains("gen_random") || !kwargs["gen_random"].cast<bool>()){
            if (!kwargs.contains("agents_path") || !kwargs.contains("tasks_path")){
                logger->log_fatal("agents_path and tasks_path must be provided if not generate instance randomly");
                exit(1);
            }
            auto agents_path = kwargs["agents_path"].cast<std::string>();
            auto tasks_path = kwargs["tasks_path"].cast<std::string>();
            agents = read_int_vec(agents_path);
            tasks = read_int_vec(tasks_path);
        } else {
            int num_agents=kwargs["num_agents"].cast<int>();
            int num_tasks=kwargs["num_tasks"].cast<int>();
            uint seed=kwargs["seed"].cast<uint>();
            gen_random_instance(grid, agents, tasks, num_agents, num_tasks, seed);
        }
        if (kwargs.contains("init_agent") && kwargs["init_agent"].cast<bool>()){
            if(!kwargs.contains("init_agent_pos")){
                std::cout<<"if init_agent, must contain init_agent_pos!" <<std::endl;
                exit(-1);
            }
            std::string init_agent_pos_str = kwargs["init_agent_pos"].cast<std::string>();
            nlohmann::json init_agent_pos_json = nlohmann::json::parse(init_agent_pos_str);
            std::vector<int> new_agents;
            for (auto & a_pos: init_agent_pos_json){
                new_agents.push_back(a_pos.get<int>());
            }
            if (new_agents.size()!= agents.size()){
                std::cout << "must have bug in init agent pos" <<std::endl;
                exit(-1);
            }
            agents.clear();
            agents = new_agents;
        }

        std::cout << agents.size() << " agents and " << tasks.size() << " tasks"<< std::endl;
        if (agents.size() > tasks.size())
            logger->log_warning("Not enough tasks for robots (number of tasks < team size)");

        // std::string task_assignment_strategy = data["taskAssignmentStrategy"].get<std::string>();
        if (task_assignment_strategy == "greedy")
        {
            system_ptr = std::make_unique<TaskAssignSystem>(grid, planner, agents, tasks, model);
        }
        else if (task_assignment_strategy == "roundrobin")
        {
            system_ptr = std::make_unique<InfAssignSystem>(grid, planner, agents, tasks, model);
        }
        else if (task_assignment_strategy == "roundrobin_fixed")
        {
            std::vector<vector<int>> assigned_tasks(agents.size());
            for (int i = 0; i < tasks.size(); i++)
            {
                assigned_tasks[i % agents.size()].push_back(tasks[i]);
            }
            system_ptr = std::make_unique<FixedAssignSystem>(grid, planner, agents, assigned_tasks, model);
        }
        else if (task_assignment_strategy == "online_generate"){
            uint seed=kwargs["seed"].cast<uint>();
            system_ptr = std::make_unique<OnlineGenerateTaskSystem>(grid, planner, agents, model, seed);
        } else if (task_assignment_strategy == "multi_category"){
            uint seed=kwargs["seed"].cast<uint>();
            system_ptr = std::make_unique<MultiCategoryTaskSystem>(grid, planner, agents, model, seed);
        } else
        {
            std::cerr << "unkown task assignment strategy " << task_assignment_strategy << std::endl;
            logger->log_fatal("unkown task assignment strategy " + task_assignment_strategy);
            exit(1);
        }
    } else {
        if (!kwargs.contains("gen_random") || !kwargs["gen_random"].cast<bool>()){
            logger->log_fatal("must generate instance randomly");
            exit(1);
        }

        int num_agents=kwargs["num_agents"].cast<int>();
        std::cout << "using kiva system (random task generation) with "<<num_agents<<" agents"<<std::endl;
        
        
        uint seed=kwargs["seed"].cast<uint>();

        std::vector<int> agents;
        std::vector<int> tasks;
        gen_random_instance(grid, agents, tasks, num_agents, 0, seed);

        if (kwargs.contains("init_agent") && kwargs["init_agent"].cast<bool>()){
            if(!kwargs.contains("init_agent_pos")){
                std::cout<<"if init_agent, must contain init_agent_pos!" <<std::endl;
                exit(-1);
            }
            std::string init_agent_pos_str = kwargs["init_agent_pos"].cast<std::string>();
            nlohmann::json init_agent_pos_json = nlohmann::json::parse(init_agent_pos_str);
            std::vector<int> new_agents;
            for (auto & a_pos: init_agent_pos_json){
                new_agents.push_back(a_pos.get<int>());
            }
            if (new_agents.size()!= agents.size()){
                std::cout << "must have bug in init agent pos" <<std::endl;
                exit(-1);
            }
            agents.clear();
            agents = new_agents;
        }
        
        system_ptr = std::make_unique<KivaSystem>(grid,planner,model,agents,seed);
    }

    
    if(kwargs.contains("dist_weights")){
        std::string weight_str=kwargs["dist_weights"].cast<std::string>();
        nlohmann::json weight_json=nlohmann::json::parse(weight_str);
        std::vector<double> weights;
        for (auto & w:weight_json) {
            weights.push_back(w.get<double>());
        }
        system_ptr->update_tasks_base_distribution(weights);
    }

    if (kwargs.contains("dist_sigma")){
        double sigma = kwargs["dist_sigma"].cast<double>();
        dist_params.sigma = sigma;
    }
    if (kwargs.contains("dist_K")){
        int K = kwargs["dist_K"].cast<int>();
        dist_params.K = K;
    }
    if(kwargs.contains("task_dist_change_interval")){
        int interval = kwargs["task_dist_change_interval"].cast<int>();
        system_ptr->task_dist_change_interval = interval;
    }

    if(kwargs.contains("task_random_type")){
        std::string random_type = kwargs["task_random_type"].cast<std::string>();
        system_ptr->set_random_type(random_type);
    }
    
    if (kwargs.contains("init_task") && kwargs["init_task"].cast<bool>()){
        if (!kwargs.contains("init_task_ids")){
            std::cout << "if use init task, must contain [init_task_ids]!"<<std::endl;
            exit(1);
        }
        std::string init_task_ids_str = kwargs["init_task_ids"].cast<std::string>();
        nlohmann::json init_task_id_json = nlohmann::json::parse(init_task_ids_str);
        std::vector<int> init_task_ids;
        for (auto & id: init_task_id_json){
            init_task_ids.push_back(id.get<int>());
        }
        system_ptr->set_init_task(true, init_task_ids);
    }

    system_ptr->set_logger(logger);
    system_ptr->set_plan_time_limit(plan_time_limit);
    system_ptr->set_preprocess_time_limit(preprocess_time_limit);

    system_ptr->set_num_tasks_reveal(num_tasks_reveal);

    // signal(SIGINT, sigint_handler);

    clock_t start_time = clock();
    system_ptr->simulate(simulation_steps);
    double runtime = (double)(clock() - start_time)/ CLOCKS_PER_SEC;

    nlohmann::json analysis=system_ptr->analyzeResults();
    analysis["cpu_runtime"] = runtime;

    // Save path if applicable
    if (kwargs.contains("save_paths") && kwargs["save_paths"].cast<bool>())
    {
        boost::filesystem::path output_dir(kwargs["file_storage_path"].cast<std::string>());
        boost::filesystem::create_directories(output_dir);
        boost::filesystem::path path_file = output_dir / "results.json";
        system_ptr->saveResults(path_file.string());
    }

    // system_ptr->saveResults("debug.json");
    return analysis.dump(4);

    // if (!vm["evaluationMode"].as<bool>())
    // {
    //     system_ptr->saveResults(vm["output"].as<std::string>());
    // }

    // delete model;
    // delete logger;
    // return 0;
}

string playground(){
	std::string json_string = R"(
	{
		"pi": 3.141,
		"happy": true
	}
	)";
	json ex1 = json::parse(json_string);

	cout << ex1["pi"] << endl;

	return ex1.dump();
}


PYBIND11_MODULE(py_driver, m) {
	// optional module docstring
    // m.doc() = ;

    m.def("playground", &playground, "Playground function to test everything");
    // m.def("add", &add, py::arg("i")=0, py::arg("j")=1);
    m.def("run", &run, "Function to run warehouse simulation");
}