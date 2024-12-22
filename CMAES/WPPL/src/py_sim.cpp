#include "py_sim.h"

PYBIND11_MODULE(py_sim, m){
    py::class_<py_sim>(m, "py_sim")
        .def(py::init<py::kwargs>())
        .def("warmup", &py_sim::warmup)
        .def("update_gg_and_step", &py_sim::update_gg_and_step)
        .def("get_curr_pos", &py_sim::get_curr_pos)
        .def("update_tasks_base_distribution", &py_sim::update_tasks_base_distribution)
        .def("get_tasks_distribution", &py_sim::get_tasks_distribution);
}

std::string py_sim::warmup(){
    this->system_ptr->total_simulation_steps = this->simulation_steps;
    this->system_ptr->warmup(this->warmup_steps);
    if (this->save_path){
        this->system_ptr->saveResults(this->path_file.string());
    }
    return this->system_ptr->analyzeResults().dump(4);
    // return "hi!";
}

std::string py_sim::update_gg_and_step(std::vector<float> edge_weights, std::vector<float> wait_costs){
    this->planner->map_weights=weight_format_conversion_with_wait_costs(this->grid, edge_weights, wait_costs);
    this->planner->update();
    int actual_sim_steps = this->system_ptr->update_gg_and_step(this->update_gg_interval);
    if (this->save_path){
        this->system_ptr->saveResults(this->path_file.string());
    }
    return this->system_ptr->analyzeCurrResults(actual_sim_steps).dump(4);
}

void py_sim::update_tasks_base_distribution(std::vector<double>& new_distribution){
    this->system_ptr->update_tasks_base_distribution(new_distribution);
}

std::vector<double> py_sim::get_tasks_distribution(){
    return this->system_ptr->get_tasks_distribution();
}

std::vector<int> py_sim::get_curr_pos(){
    auto curr_states = this->system_ptr->get_curr_states();
    std::vector<int> curr_pos;
    for (auto curr_state: curr_states){
        curr_pos.push_back(curr_state.location);
    }
    return curr_pos;
}

py_sim::py_sim(py::kwargs kwargs){
    this->simulation_steps=kwargs["simulation_steps"].cast<int>();
    this->plan_time_limit=kwargs["plan_time_limit"].cast<double>();
    this->preprocess_time_limit=kwargs["preprocess_time_limit"].cast<double>();
    this->file_storage_path=kwargs["file_storage_path"].cast<std::string>();
    this->task_assignment_strategy=kwargs["task_assignment_strategy"].cast<std::string>();
    this->num_tasks_reveal=kwargs["num_tasks_reveal"].cast<int>();

    if (kwargs.contains("warmup_steps")){
        this->warmup_steps = kwargs["warmup_steps"].cast<int>();
    }
    if (kwargs.contains("update_gg_interval")){
        this->update_gg_interval = kwargs["update_gg_interval"].cast<int>();
    }
    

    // Read in left and right weights
    if (kwargs.contains("left_w_weight")){
        this->left_w_weight = kwargs["left_w_weight"].cast<double>();
    }
    if (kwargs.contains("right_w_weight")){
        this->right_w_weight = kwargs["right_w_weight"].cast<double>();
    }

    // Read in map. Use map_path by default. If not provided, use map_json
    if (kwargs.contains("map_path")){
        std::string map_path = kwargs["map_path"].cast<std::string>();
        this->grid.load_map_from_path(map_path, left_w_weight, right_w_weight);
    } else if (kwargs.contains("map_json_str") || kwargs.contains("map_json_path")){
        json map_json;
        if (kwargs.contains("map_json_str")){
            std::string map_json_str = kwargs["map_json_str"].cast<std::string>();
            map_json = json::parse(map_json_str);
        } else {
            std::string map_json_path = kwargs["map_json_path"].cast<std::string>();
            std::ifstream f(map_json_path);
            map_json = json::parse(f);
        }
        this->grid.load_map_from_json(map_json, left_w_weight, right_w_weight);
    }

    this->logger = new Logger();
    
    this->planner = new MAPFPlanner();
    this->planner->env->map_name = grid.map_name;
    this->planner->env->file_storage_path = file_storage_path;

    if (kwargs.contains("h_update_late")){
        bool h_update_late = kwargs["h_update_late"].cast<bool>();
        this->planner->h_update_late = h_update_late;
    }

    if (kwargs.contains("config")){
        std::string config_str=kwargs["config"].cast<std::string>();
        nlohmann::json config=nlohmann::json::parse(config_str);
        this->planner->config=config;
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
            this->planner->map_weights=weight_format_conversion_with_wait_costs(grid, weights, wait_costs);
        } else {
            this->planner->map_weights=weight_format_conversion(grid, weights);
        }
    }

    this->model = new ActionModelWithRotate(grid);
    this->model->set_logger(logger);

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
            agents.clear();
            agents = this->init_agents(kwargs);
        }
        std::cout << agents.size() << " agents and " << tasks.size() << " tasks"<< std::endl;
        if (agents.size() > tasks.size())
            logger->log_warning("Not enough tasks for robots (number of tasks < team size)");

        if (this->task_assignment_strategy == "greedy"){
            this->system_ptr = std::make_unique<TaskAssignSystem>(grid, planner, agents, tasks, model);
        } else if (task_assignment_strategy == "roundrobin"){
            this->system_ptr = std::make_unique<InfAssignSystem>(grid, planner, agents, tasks, model);
        } else if (task_assignment_strategy == "roundrobin_fixed"){
            std::vector<vector<int>> assigned_tasks(agents.size());
            for (int i = 0; i < tasks.size(); i++)
            {
                assigned_tasks[i % agents.size()].push_back(tasks[i]);
            }
            this->system_ptr = std::make_unique<FixedAssignSystem>(grid, planner, agents, assigned_tasks, model);
        } else if (task_assignment_strategy == "online_generate"){
            uint seed=kwargs["seed"].cast<uint>();
            system_ptr = std::make_unique<OnlineGenerateTaskSystem>(grid, planner, agents, model, seed);
        } else if (task_assignment_strategy == "multi_category"){
            uint seed=kwargs["seed"].cast<uint>();
            system_ptr = std::make_unique<MultiCategoryTaskSystem>(grid, planner, agents, model, seed);
        } else{
            std::cerr << "unkown task assignment strategy " << task_assignment_strategy << std::endl;
            logger->log_fatal("unkown task assignment strategy " + task_assignment_strategy);
            exit(1);
        }
    } else { // warehouse map, has 'e' and 's'
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
            agents.clear();
            agents = this->init_agents(kwargs);
        }
        this->system_ptr = std::make_unique<KivaSystem>(grid,planner,model,agents,seed);
    }

    if (kwargs.contains("init_task") && kwargs["init_task"].cast<bool>()){
        std::vector<int> init_task_ids = this->init_tasks(kwargs);
        this->system_ptr->set_init_task(true, init_task_ids);
    }

    this->system_ptr->set_logger(logger);
    this->system_ptr->set_plan_time_limit(plan_time_limit);
    this->system_ptr->set_preprocess_time_limit(preprocess_time_limit);
    this->system_ptr->set_num_tasks_reveal(num_tasks_reveal);

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
    // clock_t start_time = clock();
    // system_ptr->simulate(simulation_steps);
    // double runtime = (double)(clock() - start_time)/ CLOCKS_PER_SEC;

    // nlohmann::json analysis=system_ptr->analyzeResults();
    // analysis["cpu_runtime"] = runtime;

    // // Save path if applicable
    if (kwargs.contains("save_paths") && kwargs["save_paths"].cast<bool>())
    {
        this->save_path = true;
        boost::filesystem::path output_dir(kwargs["file_storage_path"].cast<std::string>());
        boost::filesystem::create_directories(output_dir);
        this->path_file = output_dir / "results.json";
    //     system_ptr->saveResults(path_file.string());
    }

    // // system_ptr->saveResults("debug.json");
    // return analysis.dump(4);
}

std::vector<int> py_sim::init_agents(py::kwargs& kwargs){
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
    return new_agents;
}

std::vector<int> py_sim::init_tasks(py::kwargs& kwargs){
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
    return init_task_ids;
}