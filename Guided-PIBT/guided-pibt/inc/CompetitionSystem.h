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
#include <random>
#include "TaskDistGenerator.h"

class BaseSystem{
public:


	BaseSystem(Grid &grid, MAPFPlanner* planner, ActionModelWithRotate* model):
        map(grid), planner(planner), env(planner->env), model(model)
    {}

	virtual ~BaseSystem(){

        //safely exit: wait for join the thread then delete planner and exit
        if (started){
            task_td.join();
        }
        if (planner != nullptr){
            delete planner;
        }
    };

	void simulate(int simulation_time);
    void warmup(int warmup_time);
    void update_gg_and_step(std::vector<double> new_map_weights);
    void update_map_weights(std::vector<double> map_weights);

    std::string action2symbol(int action) const;


    void savePaths(const string &fileName, int option) const; //option = 0: save actual movement, option = 1: save planner movement
    //void saveSimulationIssues(const string &fileName) const;
    void saveResults(const string &fileName) const;
    nlohmann::json analyzeResults(int simulation_time) const;
    nlohmann::json analyzeCurrResults(int summarize_interval) const;

    int num_tasks_reveal = 1;

    void set_num_tasks_reveal(int num){num_tasks_reveal = num;};
    void set_plan_time_limit(int limit){plan_time_limit = limit;};
    void set_preprocess_time_limit(int limit){preprocess_time_limit = limit;};

    vector<Action> plan();
    vector<Action> plan_wrapper();

    Logger* logger = nullptr;
    void set_logger(Logger* logger){this->logger = logger;}

    int task_dist_change_interval = -1;
    std::string random_type;

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

    
    int warmup_time;
    int update_gg_interval;
    int simulation_time;

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

    bool has_timeout = false;

    std::vector<Path> paths;
    std::vector<std::list<Task > > finished_tasks; // location + finish time

    vector<State> starts;
    int num_of_agents;

    vector<State> curr_states;

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


	void initialize();
    bool planner_initialize();
	virtual void update_tasks() = 0;

    void update_env_traffic_usage();
    void sync_shared_env();

    list<Task> move(vector<Action>& actions);
    bool valid_moves(vector<State>& prev, vector<Action>& next);


    void log_preprocessing(bool succ);
    void log_event_assigned(int agent_id, int task_id, int task_loc, int timestep);
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
        if (start_locs.size() != tasks.size()){
            std::cerr << "agent num does not match the task assignment" << std::endl;
            exit(1);
        }

        int task_id = 0;
        num_of_agents = start_locs.size();
        starts.resize(num_of_agents);
        task_queue.resize(num_of_agents);
        for (size_t i = 0; i < start_locs.size(); i++){
            starts[i] = State(start_locs[i], 0, 0);
            for (auto& task_location: tasks[i]){
                all_tasks.emplace_back(task_id++, task_location, 0, (int)i);
                task_queue[i].emplace_back(all_tasks.back().task_id, all_tasks.back().location, all_tasks.back().t_assigned, all_tasks.back().agent_assigned);
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
        for (auto& task_location: tasks){
            all_tasks.emplace_back(task_id++, task_location);
            task_queue.emplace_back(all_tasks.back().task_id, all_tasks.back().location);
            //task_queue.emplace_back(task_id++, task_location);
        }
        num_of_agents = start_locs.size();
        starts.resize(num_of_agents);
        for (size_t i = 0; i < start_locs.size(); i++){
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

        for (size_t i = 0; i < start_locs.size(); i++){
            if (grid.map[start_locs[i]] == 1)
            {
                cout<<"error: agent "<<i<<"'s start location is an obstacle("<<start_locs[i]<<")"<<endl;
                exit(0);
            }
            starts[i] = State(start_locs[i], 0, -1);
        }
        for (size_t i=0; i<tasks.size(); ++i){
            if (grid.map[tasks[i]] == 1){
                cout<<"error: task "<<i<<"'s location is an obstacle("<<tasks[i]<<")"<<endl;
                exit(0);
            }
        }
    };


	~InfAssignSystem(){};

private:
    std::vector<int>& tasks;
    std::vector<int> task_counter;
    int tasks_size;
    int task_id = 0;

	void update_tasks();

};

class KivaSystem: public BaseSystem 
{
public:
    KivaSystem(Grid &grid, MAPFPlanner* planner, ActionModelWithRotate* model, int num_agents, uint seed):
        BaseSystem(grid, planner, model), MT(seed), task_id(0)
    {
        num_of_agents = num_agents;
        starts.resize(num_of_agents);

        std::shuffle(grid.empty_locs.begin(),grid.empty_locs.end(), MT);

        for (size_t i = 0; i < num_of_agents; i++)
        {
            starts[i] = State(grid.empty_locs[i], 0, -1);
            prev_task_locs.push_back(grid.empty_locs[i]);
        }

        // Initialize agent home location (workstation) distribution

        this->home_loc_weights = grid.agent_home_loc_weights;

        this->end_pts_weights = std::vector<double>(grid.end_points.size(), 1.0);

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

    ~KivaSystem(){};

    void random_update_tasks_distribution() override;
    // std::vector<double> get_tasks_distribution() override;

private:
    std::mt19937 MT;
    int task_id=0;

    void update_tasks();
    std::discrete_distribution<int> agent_home_loc_dist;
    std::discrete_distribution<int> agent_end_pts_dist;

    std::vector<double> home_loc_weights;
    std::vector<double> end_pts_weights;

    std::vector<int> prev_task_locs;

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
        this->empty_weights.resize(this->map.empty_locs.size(), 1.0);
        
        this->goal_loc_dist = std::discrete_distribution<int>(
            this->empty_weights.begin(), this->empty_weights.end()
        );
    }
    void random_update_tasks_distribution() override;

    private:
    std::mt19937 MT;
    std::vector<double> empty_weights;
    std::discrete_distribution<int> goal_loc_dist;
    int task_id = 0;
    void update_tasks();
};

