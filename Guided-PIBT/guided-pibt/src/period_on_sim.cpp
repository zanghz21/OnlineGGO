#include "period_on_sim.h"

// cpp simulator for [p-on]+GPIBT
PYBIND11_MODULE(period_on_sim, m){
    py::class_<period_on_sim>(m, "period_on_sim")
        .def(py::init<py::kwargs>())
        .def("warmup", &period_on_sim::warmup)
        .def("update_gg_and_step", &period_on_sim::update_gg_and_step);
}


std::string period_on_sim::warmup(){
    std::cout << "in warmup, map weights size = "<< this->planner->env->map_weights.size()<< std::endl;
    this->py_system_ptr->warmup(this->warmup_time);
    nlohmann::json result_json = this->py_system_ptr->analyzeCurrResults(this->warmup_time);
    return result_json.dump(4);
}

std::string period_on_sim::update_gg_and_step(std::vector<double> map_weights){
    // clear system h
    // update map_weights
    // recompute system h
    // step
    // save_result
    this->py_system_ptr->update_gg_and_step(map_weights);
    nlohmann::json result_json = this->py_system_ptr->analyzeCurrResults(this->update_gg_interval);
    return result_json.dump(4);
}

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


period_on_sim::period_on_sim(py::kwargs kwargs){
    int plan_time_limit = 10;
    Logger* logger = new Logger();
    this->planner = new MAPFPlanner();

    std::string inputFilePath = "";

    bool gen_tasks = false;
    if (kwargs.contains("gen_tasks")){
        gen_tasks = kwargs["gen_tasks"].cast<bool>();
    }

    std::string map_path;
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
        this->agents = read_int_vec(base_folder + read_param_json<std::string>(data, "agentFile"),team_size);
        this->tasks = read_int_vec(base_folder + read_param_json<std::string>(data, "taskFile"));
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
        this->grid.load_from_path(map_path);
        // std::cout << grid.agent_home_locs.size() <<", "<< grid.sortation_points.size() <<", "<< grid.end_points.size()<<std::endl;
        if (grid.agent_home_locs.size() == 0){
            gen_random_instance(this->grid, this->agents, this->tasks, num_agents, num_tasks, seed);
        }
    }
    
    // TODO: should rewrite here!!!
    this->planner->env->map_name = map_path.substr(map_path.find_last_of("/") + 1);
    this->planner->env->file_storage_path = "";

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
        double min_map_weight = 100000; // todo: change to double max
        for (auto & w:weight_json) {
            double weight = w.get<double>();
            map_weights.push_back(w.get<double>());
            if (weight < min_map_weight){
                min_map_weight = weight;
            }
        }
        this->planner->env->map_weights = map_weights;
        // this->planner->env->min_map_weight = min_map_weight;
    } else {
        this->planner->env->map_weights = std::vector<double>(this->grid.cols * this->grid.rows * 4, 1.0);
    }
    std::cout << "map weights size = "<< this->planner->env->map_weights.size()<<std::endl;

    ActionModelWithRotate* model = new ActionModelWithRotate(grid);
    // model->set_logger(logger);

    if (grid.agent_home_locs.size() == 0){
        if (task_assignment_strategy=="greedy"){
            this->py_system_ptr = std::make_unique<TaskAssignSystem>(this->grid, this->planner, this->agents, this->tasks, model);
        } else if (task_assignment_strategy=="roundrobin"){
            this->py_system_ptr = std::make_unique<InfAssignSystem>(this->grid, this->planner, this->agents, this->tasks, model);
        } else if (task_assignment_strategy=="roundrobin_fixed"){
            std::vector<vector<int>> assigned_tasks(this->agents.size());
            for(int i = 0; i < this->tasks.size(); i++){
                assigned_tasks[i%this->agents.size()].push_back(this->tasks[i]);
            }
            this->py_system_ptr = std::make_unique<FixedAssignSystem>(this->grid, this->planner, this->agents, assigned_tasks, model);
        } else if (task_assignment_strategy == "online_generate"){
            uint seed=kwargs["seed"].cast<uint>();
            this->py_system_ptr = std::make_unique<OnlineGenerateTaskSystem>(this->grid, this->planner, this->agents, model, seed);
        } else{
            std::cerr << "unknown task assignment strategy " << task_assignment_strategy << std::endl;
            logger->log_fatal("unknown task assignment strategy " + task_assignment_strategy);
            exit(1);
        }
    } else {
        int num_agents=kwargs["num_agents"].cast<int>();
        std::cout << "using kiva system (random task generation) with "<<num_agents<<" agents"<<std::endl;
        
        uint seed=kwargs["seed"].cast<uint>();
        this->py_system_ptr = std::make_unique<KivaSystem>(this->grid,this->planner,model,num_agents,seed);
    }

    if(kwargs.contains("task_dist_change_interval")){
        int interval = kwargs["task_dist_change_interval"].cast<int>();
        this->py_system_ptr->task_dist_change_interval = interval;
    }

    if(kwargs.contains("task_random_type")){
        std::string random_type = kwargs["task_random_type"].cast<std::string>();
        this->py_system_ptr->set_random_type(random_type);
    }

    this->py_system_ptr->set_plan_time_limit(plan_time_limit);
    this->py_system_ptr->set_preprocess_time_limit(INT_MAX);

    this->py_system_ptr->set_num_tasks_reveal(num_tasks_reveal);

    // signal(SIGINT, sigint_handler);
    if (kwargs.contains("simu_time")){
        this->simuTime = kwargs["simu_time"].cast<int>();
    }

    if (simuTime == -1){
        this->simuTime = (this->grid.cols + this->grid.rows)/2 * 10;
    }
    this->py_system_ptr->simulation_time = this->simuTime;

    this->warmup_time = kwargs["warmup_time"].cast<int>();
    this->py_system_ptr->warmup_time = this->warmup_time;
    
    this->update_gg_interval = kwargs["update_gg_interval"].cast<int>();
    this->py_system_ptr->update_gg_interval = this->update_gg_interval;
    // std::cout << "start simulation!" <<std::endl;
    // py_system_ptr->simulate(simuTime);
    // // py_system_ptr->saveResults("./exp/test.json");
    // nlohmann::json result_json = py_system_ptr->analyzeResults(simuTime);
    
    // bool save_result = false;
    // std::string save_path = "";
    // if (kwargs.contains("save_path")){
    //     save_result = true;
    //     save_path = kwargs["save_path"].cast<std::string>();
    // }
    
    // if (save_result){
    //     py_system_ptr->saveResults(save_path);
    // }
    // return result_json.dump(4);
}