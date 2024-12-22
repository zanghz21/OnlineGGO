#pragma once

// #include "BasicSystem.h"
#include "SharedEnv.h"
#include "Grid.h"
#include "Tasks.h"
#include "ActionModel.h"
#include "MAPFPlanner.h"
#include "Logger.h"
#include <pthread.h>
#include <future>
#include <cstdlib>
#ifdef MAP_OPT
#include "nlohmann/json.hpp"
#include "util/analyze.h"
#include "util/TaskDistGenerator.h"
#endif

#ifndef NO_ROT

class BaseSystem
{
public:
    int num_tasks_reveal = 1;
    Logger* logger = nullptr;

	BaseSystem(Grid &grid, MAPFPlanner* planner, ActionModelWithRotate* model):
        map(grid), planner(planner), env(planner->env), model(model)
    {}

	virtual ~BaseSystem()
    {
        //safely exit: wait for join the thread then delete planner and exit
        if (started)
        {
            task_td.join();
        }
        if (planner != nullptr)
        {
            delete planner;
        }
    };

    void set_num_tasks_reveal(int num){num_tasks_reveal = num;};
    void set_plan_time_limit(int limit){plan_time_limit = limit;};
    void set_preprocess_time_limit(int limit){preprocess_time_limit = limit;};
    void set_logger(Logger* logger){this->logger = logger;}
    void set_init_task(bool init_task, std::vector<int> init_task_ids){
        this->init_task = init_task;
        this->init_task_ids = init_task_ids;
    }

	void simulate(int simulation_time);
    vector<Action> plan();
    vector<Action> plan_wrapper();

    void savePaths(const string &fileName, int option) const; //option = 0: save actual movement, option = 1: save planner movement
    //void saveSimulationIssues(const string &fileName) const;
    void saveResults(const string &fileName) const;

#ifdef MAP_OPT
    nlohmann::json analyzeResults();
#endif


protected:
    Grid map;

    std::future<std::vector<Action>> future;
    std::thread task_td;
    bool started = false;

    MAPFPlanner* planner;
    SharedEnvironment* env;

    ActionModelWithRotate* model;

    // #timesteps for simulation
    int timestep;

    int preprocess_time_limit=10;

    int plan_time_limit = 3;

    std::vector<Path> paths;
    std::vector<std::list<Task > > finished_tasks; // location + finish time

    vector<State> starts;
    int num_of_agents;

    vector<State> curr_states;

    
    vector<list<State>> execution_paths;
    vector<list<State>> planning_paths;
    
    vector<list<Action>> actual_movements;
    vector<list<Action>> planner_movements;

    // tasks that haven't been finished but have been revealed to agents;
    vector< deque<Task > > assigned_tasks;

    vector<list<std::tuple<int,int,std::string>>> events;
    list<Task> all_tasks;

    //for evaluation
    vector<int> solution_costs;
    int num_of_task_finish = 0;
    list<double> planner_times; 
    bool fast_mover_feasible = true;

    bool init_task = false;
    std::vector<int> init_task_ids;


	void initialize();
    bool planner_initialize();
	virtual void update_tasks() = 0;

    void sync_shared_env();

    list<Task> move(vector<Action>& actions);
    bool valid_moves(vector<State>& prev, vector<Action>& next);

    void log_preprocessing(bool succ);
    void log_event_assigned(int agent_id, int task_id, int timestep);
    void log_event_finished(int agent_id, int task_id, int timestep);


};


class FixedAssignSystem : public BaseSystem
{
public:
	FixedAssignSystem(Grid &grid, string agent_task_filename, MAPFPlanner* planner, ActionModelWithRotate *model):
        BaseSystem(grid, planner, model)
    {
        load_agent_tasks(agent_task_filename);
    };

	FixedAssignSystem(Grid &grid, MAPFPlanner* planner, std::vector<int>& start_locs, std::vector<vector<int>>& tasks, ActionModelWithRotate* model):
        BaseSystem(grid, planner, model)
    {
        if (start_locs.size() != tasks.size())
        {
            std::cerr << "agent num does not match the task assignment" << std::endl;
            exit(1);
        }

        int task_id = 0;
        num_of_agents = start_locs.size();
        starts.resize(num_of_agents);
        task_queue.resize(num_of_agents);
        for (size_t i = 0; i < start_locs.size(); i++)
        {
            starts[i] = State(start_locs[i], 0, 0);
            for (auto& task_location: tasks[i])
            {
                all_tasks.emplace_back(task_id++, task_location, 0, (int)i);
                task_queue[i].emplace_back(all_tasks.back().task_id, all_tasks.back().location, 
                    all_tasks.back().t_assigned, all_tasks.back().agent_assigned);
            }
            // task_queue[i] = deque<int>(tasks[i].begin(), tasks[i].end());
        }
    };

	~FixedAssignSystem(){};

    bool load_agent_tasks(string fname);


private:
    vector<deque<Task>> task_queue;

	void update_tasks();
};


class TaskAssignSystem : public BaseSystem
{
public:
	TaskAssignSystem(Grid &grid, MAPFPlanner* planner, std::vector<int>& start_locs, std::vector<int>& tasks, ActionModelWithRotate* model):
        BaseSystem(grid, planner, model)
    {
        int task_id = 0;
        for (auto& task_location: tasks)
        {
            all_tasks.emplace_back(task_id++, task_location);
            task_queue.emplace_back(all_tasks.back().task_id, all_tasks.back().location);
            //task_queue.emplace_back(task_id++, task_location);
        }
        num_of_agents = start_locs.size();
        starts.resize(num_of_agents);
        for (size_t i = 0; i < start_locs.size(); i++)
        {
            starts[i] = State(start_locs[i], 0, 0);
        }
    };

	~TaskAssignSystem(){};


private:
    deque<Task> task_queue;

	void update_tasks();
};


class InfAssignSystem : public BaseSystem
{
public:
	InfAssignSystem(Grid &grid, MAPFPlanner* planner, std::vector<int>& start_locs, std::vector<int>& tasks, ActionModelWithRotate* model):
        tasks(tasks), BaseSystem(grid, planner, model)
    {
        num_of_agents = start_locs.size();
        starts.resize(num_of_agents);
        task_counter.resize(num_of_agents,0);
        tasks_size = tasks.size();

        for (size_t i = 0; i < start_locs.size(); i++)
        {
            if (grid.map[start_locs[i]] == 1)
            {
                cout<<"error: agent "<<i<<"'s start location is an obstacle("<<start_locs[i]<<")"<<endl;
                exit(0);
            }
            starts[i] = State(start_locs[i], 0, 0);
        }

        // char * LOAD_FP=std::getenv("LOAD_FP");
        // std::cerr<<"LOAD_FP: "<<LOAD_FP<<std::endl;
        // if (LOAD_FP!=NULL && strlen(LOAD_FP)!=0) {
        //     resume_from_file(LOAD_FP, grid.cols);
        // }

    };

	~InfAssignSystem(){};


private:
    std::vector<int> tasks;
    std::vector<int> task_counter;
    int tasks_size;
    int task_id = 0;

	void update_tasks();

    void resume_from_file(string fname, int w);
};

#else

class BaseSystem
{
public:
    int num_tasks_reveal = 1;
    int task_dist_change_interval = -1;

    Logger* logger = nullptr;

	BaseSystem(Grid &grid, MAPFPlanner* planner, ActionModelWithRotate* model):
        map(grid), planner(planner), env(planner->env), model(model)
    {}

	virtual ~BaseSystem()
    {
        //safely exit: wait for join the thread then delete planner and exit
        if (started)
        {
            task_td.join();
        }
        if (planner != nullptr)
        {
            delete planner;
        }
    };

    void set_num_tasks_reveal(int num){num_tasks_reveal = num;};
    void set_plan_time_limit(int limit){plan_time_limit = limit;};
    void set_preprocess_time_limit(int limit){preprocess_time_limit = limit;};
    void set_logger(Logger* logger){this->logger = logger;}
    void set_init_task(bool init_task, std::vector<int> init_task_ids){
        this->init_task = init_task;
        this->init_task_ids = init_task_ids;
    }

	void simulate(int simulation_time);

    void warmup(int total_warmup_steps);
    int update_gg_and_step(int update_gg_interval);
    // void update_gg_and_step(int update_interval);

    vector<Action> plan();
    vector<Action> plan_wrapper();

    void savePaths(const string &fileName, int option) const; //option = 0: save actual movement, option = 1: save planner movement
    //void saveSimulationIssues(const string &fileName) const;
    void saveResults(const string &fileName) const;

    std::string random_type;
    virtual void update_tasks_base_distribution(std::vector<double> new_distribution){
        std::cout << "in BaseSystem::update_tasks_base_distribution "<<std::endl;
        exit(1);
    };

    virtual void update_tasks_distribution(){
        std::cout << "in BaseSystem::update_tasks_distribution "<<std::endl;
        exit(1);
    };

    virtual void random_update_tasks_distribution(){
        std::cout << "in BaseSystem::random_update_tasks_distribution "<<std::endl;
        exit(1);
    };

    virtual std::vector<double> get_tasks_distribution(){
        std::cout << "in BaseSystem::get_tasks_distribution" <<std::endl;
        exit(1);
    }

    void set_random_type(std::string _random_type){
        this->random_type = _random_type;
    }

#ifdef MAP_OPT
    nlohmann::json analyzeResults();
    nlohmann::json analyzeCurrResults(int update_gg_interval);
    int total_simulation_steps;
    const vector<State>& get_curr_states() const {
        return this->curr_states;
    }
#endif

protected:
    Grid map;

    std::future<std::vector<Action>> future;
    std::thread task_td;
    bool started = false;

    MAPFPlanner* planner;
    SharedEnvironment* env;

    ActionModelWithRotate* model;

    int warmupstep=0;
    // #timesteps for simulation
    int timestep;

    int preprocess_time_limit=10;

    int plan_time_limit = 3;

    std::vector<Path> paths;
    std::vector<std::list<Task > > finished_tasks; // location + finish time
    std::vector<int> curr_finish_task_agents;


    vector<State> starts;
    int num_of_agents;

    vector<State> curr_starts;
    vector<State> curr_states;

    vector<list<State>> execution_paths;
    vector<list<State>> planning_paths;
    vector<list<Action>> actual_movements;
    vector<list<Action>> planner_movements;

    // tasks that haven't been finished but have been revealed to agents;
    vector< deque<Task > > assigned_tasks;

    vector<list<std::tuple<int,int,std::string>>> events;
    list<Task> all_tasks;

    //for evaluation
    vector<int> solution_costs;
    int num_of_task_finish = 0;
    list<double> planner_times; 
    bool fast_mover_feasible = true;

    bool init_task = false;
    std::vector<int> init_task_ids;

	void initialize();
    bool planner_initialize();
	virtual void update_tasks() = 0;

    void sync_shared_env();

    list<Task> move(vector<Action>& actions);
    bool valid_moves(vector<State>& prev, vector<Action>& next);

    void log_preprocessing(bool succ);
    void log_event_assigned(int agent_id, int task_id, int timestep);
    void log_event_finished(int agent_id, int task_id, int timestep);

};


class FixedAssignSystem : public BaseSystem
{
public:
	FixedAssignSystem(Grid &grid, string agent_task_filename, MAPFPlanner* planner, ActionModelWithRotate *model):
        BaseSystem(grid, planner, model)
    {
        load_agent_tasks(agent_task_filename);
    };

	FixedAssignSystem(Grid &grid, MAPFPlanner* planner, std::vector<int>& start_locs, std::vector<vector<int>>& tasks, ActionModelWithRotate* model):
        BaseSystem(grid, planner, model)
    {
        if (start_locs.size() != tasks.size())
        {
            std::cerr << "agent num does not match the task assignment" << std::endl;
            exit(1);
        }

        int task_id = 0;
        num_of_agents = start_locs.size();
        starts.resize(num_of_agents);
        task_queue.resize(num_of_agents);
        for (size_t i = 0; i < start_locs.size(); i++)
        {
            starts[i] = State(start_locs[i], 0, -1);
            for (auto& task_location: tasks[i])
            {
                all_tasks.emplace_back(task_id++, task_location, 0, (int)i);
                task_queue[i].emplace_back(all_tasks.back().task_id, all_tasks.back().location, 
                    all_tasks.back().t_assigned, all_tasks.back().agent_assigned);
            }
            // task_queue[i] = deque<int>(tasks[i].begin(), tasks[i].end());
        }
    };

	~FixedAssignSystem(){};

    bool load_agent_tasks(string fname);


private:
    vector<deque<Task>> task_queue;

	void update_tasks();
};


class TaskAssignSystem : public BaseSystem
{
public:
	TaskAssignSystem(Grid &grid, MAPFPlanner* planner, std::vector<int>& start_locs, std::vector<int>& tasks, ActionModelWithRotate* model):
        BaseSystem(grid, planner, model)
    {
        int task_id = 0;
        for (auto& task_location: tasks)
        {
            all_tasks.emplace_back(task_id++, task_location);
            task_queue.emplace_back(all_tasks.back().task_id, all_tasks.back().location);
            //task_queue.emplace_back(task_id++, task_location);
        }
        num_of_agents = start_locs.size();
        starts.resize(num_of_agents);
        for (size_t i = 0; i < start_locs.size(); i++)
        {
            starts[i] = State(start_locs[i], 0, -1);
        }
    };

	~TaskAssignSystem(){};


private:
    deque<Task> task_queue;

	void update_tasks();
};


class InfAssignSystem : public BaseSystem
{
public:
	InfAssignSystem(Grid &grid, MAPFPlanner* planner, std::vector<int>& start_locs, std::vector<int>& tasks, ActionModelWithRotate* model):
        tasks(tasks), BaseSystem(grid, planner, model)
    {
        num_of_agents = start_locs.size();
        starts.resize(num_of_agents);
        task_counter.resize(num_of_agents,0);
        tasks_size = tasks.size();

        for (size_t i = 0; i < start_locs.size(); i++)
        {
            if (grid.map[start_locs[i]] == 1)
            {
                cout<<"error: agent "<<i<<"'s start location is an obstacle("<<start_locs[i]<<")"<<endl;
                exit(0);
            }
            starts[i] = State(start_locs[i], 0, -1);
        }

        // char * LOAD_FP=std::getenv("LOAD_FP");
        // std::cerr<<"LOAD_FP: "<<LOAD_FP<<std::endl;
        // if (LOAD_FP!=NULL && strlen(LOAD_FP)!=0) {
        //     resume_from_file(LOAD_FP, grid.cols);
        // }


    };

	~InfAssignSystem(){};


private:
    std::vector<int> tasks;
    std::vector<int> task_counter;
    int tasks_size;
    int task_id = 0;

	void update_tasks();

    void resume_from_file(string fname, int w);

};


class KivaSystem: public BaseSystem 
{
public:
    KivaSystem(Grid &grid, MAPFPlanner* planner, ActionModelWithRotate* model, std::vector<int>& start_locs, uint seed):
        BaseSystem(grid, planner, model), MT(seed), task_id(0)
    {
        num_of_agents = start_locs.size();
        starts.resize(num_of_agents);

        this->random_type = "LR";

        for (size_t i = 0; i < start_locs.size(); i++)
        {
            if (grid.map[start_locs[i]] == 1)
            {
                cout<<"error: agent "<<i<<"'s start location is an obstacle("<<start_locs[i]<<")"<<endl;
                exit(0);
            }
            starts[i] = State(start_locs[i], 0, -1);
            prev_task_locs.push_back(start_locs[i]);
        }
        // std::shuffle(grid.empty_locations.begin(),grid.empty_locations.end(), MT);

        // for (size_t i = 0; i < num_of_agents; i++)
        // {
        //     starts[i] = State(grid.empty_locations[i], 0, -1);
        //     prev_task_locs.push_back(grid.empty_locations[i]);
        // }

        // Initialize agent home location (workstation) distribution

        this->home_loc_weights.clear();
        for (auto w: grid.agent_home_loc_weights){
            this->home_loc_weights.push_back(w);
        }

        this->end_pts_weights.clear();
        for (int i=0; i<grid.end_points.size(); ++i){
            this->end_pts_weights.push_back(1.0);
        }

        this->agent_home_loc_dist = std::discrete_distribution<int>(
            this->home_loc_weights.begin(),
            this->home_loc_weights.end()
        );
        
        this->agent_end_pts_dist = std::discrete_distribution<int>(
            this->end_pts_weights.begin(),
            this->end_pts_weights.end()
        );
        
        cout << "agent_home_loc distribution: ";
        for (auto w: grid.agent_home_loc_weights)
        {
            cout << w << ", ";
        }
        cout << endl;

    };

    void update_tasks_base_distribution(std::vector<double> new_distribution) override;
    void update_tasks_distribution() override;
    void random_update_tasks_distribution() override;
    std::vector<double> get_tasks_distribution() override;
private:
    std::mt19937 MT;
    int task_id=0;

    void update_tasks();
    std::discrete_distribution<int> agent_home_loc_dist;
    std::discrete_distribution<int> agent_end_pts_dist;

    std::vector<int> prev_task_locs;
    std::vector<double> home_loc_weights;
    std::vector<double> end_pts_weights;

};

class OnlineGenerateTaskSystem: public BaseSystem{
    public:
    OnlineGenerateTaskSystem(Grid &grid, MAPFPlanner* planner, std::vector<int>& start_locs, ActionModelWithRotate* model, uint seed):
        BaseSystem(grid, planner, model), MT(seed)
    {
        this->num_of_agents = start_locs.size();
        this->starts.resize(this->num_of_agents);

        for (size_t i = 0; i < start_locs.size(); i++){
            if (grid.map[start_locs[i]] == 1)
            {
                cout<<"error: agent "<<i<<"'s start location is an obstacle("<<start_locs[i]<<")"<<endl;
                exit(0);
            }
            starts[i] = State(start_locs[i], 0, -1);
        }

        this->empty_weights.clear();
        this->random_type = "uniform"; // we can change this after initialization
        this->empty_weights.resize(this->map.empty_locations.size(), 1.0);
        
        this->goal_loc_dist = std::discrete_distribution<int>(
            this->empty_weights.begin(), this->empty_weights.end()
        );
    }
    void random_update_tasks_distribution() override;
    std::vector<double> get_tasks_distribution() override;
    
    private:
    std::mt19937 MT;
    std::vector<double> empty_weights;
    std::discrete_distribution<int> goal_loc_dist;
    int task_id = 0;
    void update_tasks();
};

class MultiCategoryTaskSystem: public BaseSystem{
    // for 7:2:1 task distribution strategy
    public:
    MultiCategoryTaskSystem(Grid &grid, MAPFPlanner* planner, std::vector<int>& start_locs, ActionModelWithRotate* model, uint seed):
        BaseSystem(grid, planner, model), MT(seed)
    {
        this->num_of_agents = start_locs.size();
        this->starts.resize(this->num_of_agents);

        for (size_t i = 0; i < start_locs.size(); i++){
            if (grid.map[start_locs[i]] == 1)
            {
                cout<<"error: agent "<<i<<"'s start location is an obstacle("<<start_locs[i]<<")"<<endl;
                exit(0);
            }
            starts[i] = State(start_locs[i], 0, -1);
        }

        this->random_update_tasks_distribution();
    }
    void random_update_tasks_distribution() override;
    std::vector<double> get_tasks_distribution() override;
    private:
    std::mt19937 MT;
    std::vector<double> empty_weights;
    std::discrete_distribution<int> goal_loc_dist;
    int task_id = 0;
    void update_tasks();
};
#endif