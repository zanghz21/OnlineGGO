#include <cmath>
#include "CompetitionSystem.h"
#include <boost/tokenizer.hpp>
#include "nlohmann/json.hpp"
#include <functional>
#include <Logger.h>
#include "util/Timer.h"
#include "util/Analyzer.h"
#include "util/MyLogger.h"

using json = nlohmann::ordered_json;

#ifndef NO_ROT

list<Task> BaseSystem::move(vector<Action>& actions)
{
    // actions.resize(num_of_agents, Action::NA);
    for (int k = 0; k < num_of_agents; k++)
    {
        //log->log_plan(false,k);
        if (k >= actions.size()){
            fast_mover_feasible = false;
            planner_movements[k].push_back(Action::NA);
        }
        else
        {
            planner_movements[k].push_back(actions[k]);
        }
    }

    list<Task> finished_tasks_this_timestep; // <agent_id, task_id, timestep>
    if (!valid_moves(curr_states, actions))
    {
        fast_mover_feasible = false;
        actions = std::vector<Action>(num_of_agents, Action::W);
    }

    curr_states = model->result_states(curr_states, actions);
    // agents do not move
    for (int k = 0; k < num_of_agents; k++)
    {
        if (!assigned_tasks[k].empty() && curr_states[k].location == assigned_tasks[k].front().location)
        {
            Task task = assigned_tasks[k].front();
            assigned_tasks[k].pop_front();
            task.t_completed = timestep;
            finished_tasks_this_timestep.push_back(task);
            events[k].push_back(make_tuple(task.task_id, timestep,"finished"));
            log_event_finished(k, task.task_id, timestep);
        }
        paths[k].push_back(curr_states[k]);
        actual_movements[k].push_back(actions[k]);
    }

    return finished_tasks_this_timestep;
}


// This function might not work correctly with small map (w or h <=2)
bool BaseSystem::valid_moves(vector<State>& prev, vector<Action>& action)
{
    return model->is_valid(prev, action);
}


void BaseSystem::sync_shared_env() {
    env->goal_locations.resize(num_of_agents);
    for (size_t i = 0; i < num_of_agents; i++)
    {
        env->goal_locations[i].clear();
        for (auto& task: assigned_tasks[i])
        {
            env->goal_locations[i].push_back({task.location, task.t_assigned });
        }
    }
    env->curr_timestep = timestep;
    env->curr_states = curr_states;
}


vector<Action> BaseSystem::plan_wrapper()
{
    // std::cout<<"wrapper called"<<std::endl;
    vector<Action> actions;
    // std::cout<<"planning"<<std::endl;
    
    vector<list<State>> cur_exec_paths(num_of_agents);
    vector<list<State>> cur_plan_paths(num_of_agents);

    planner->plan(plan_time_limit, actions, cur_exec_paths, cur_plan_paths);
    this->execution_paths = cur_exec_paths;
    this->planning_paths = cur_plan_paths;

    return actions;
}


vector<Action> BaseSystem::plan()
{
    return plan_wrapper();

    // using namespace std::placeholders;
    // if (started && future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    // {
    //     std::cout << started << "     " << (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) << std::endl;
    //     if(logger)
    //     {
    //         logger->log_info("planner cannot run because the previous run is still running", timestep);
    //     }

    //     if (future.wait_for(std::chrono::seconds(plan_time_limit)) == std::future_status::ready)
    //     {
    //         task_td.join();
    //         started = false;
    //         return future.get();
    //     }
    //     logger->log_info("planner timeout", timestep);
    //     return {};
    // }

    // std::packaged_task<std::vector<Action>()> task(std::bind(&BaseSystem::plan_wrapper, this));
    // future = task.get_future();
    // if (task_td.joinable())
    // {
    //     task_td.join();
    // }
    // task_td = std::thread(std::move(task));
    // started = true;
    // if (future.wait_for(std::chrono::seconds(plan_time_limit)) == std::future_status::ready)
    // {
    //     task_td.join();
    //     started = false;
    //     return future.get();
    // }
    // logger->log_info("planner timeout", timestep);
    // return {};
}


bool BaseSystem::planner_initialize()
{
    planner->initialize(preprocess_time_limit);
    return true;

    // using namespace std::placeholders;
    // std::packaged_task<void(int)> init_task(std::bind(&MAPFPlanner::initialize, planner, std::placeholders::_1));
    // auto init_future = init_task.get_future();

    // auto init_td = std::thread(std::move(init_task), preprocess_time_limit);
    // if (init_future.wait_for(std::chrono::seconds(preprocess_time_limit)) == std::future_status::ready)
    // {
    //     init_td.join();
    //     return true;
    // }

    // init_td.detach();
    // return false;
}


void BaseSystem::log_preprocessing(bool succ)
{
    if (logger == nullptr)
        return;
    if (succ)
    {
        logger->log_info("Preprocessing success", timestep);
    } 
    else
    {
        logger->log_fatal("Preprocessing timeout", timestep);
    }
}


void BaseSystem::log_event_assigned(int agent_id, int task_id, int timestep)
{
    // logger->log_info("Task " + std::to_string(task_id) + " is assigned to agent " + std::to_string(agent_id), timestep);
}


void BaseSystem::log_event_finished(int agent_id, int task_id, int timestep) 
{
    // logger->log_info("Agent " + std::to_string(agent_id) + " finishes task " + std::to_string(task_id), timestep);
}


void BaseSystem::simulate(int simulation_time)
{
    //init logger
    //Logger* log = new Logger();

    ONLYDEV(g_timer.record_p("simulate_start");)

    initialize();

    ONLYDEV(g_timer.record_d("simulate_start","initialize_end","initialization");)
    int num_of_tasks = 0;

    for (; timestep < simulation_time; )
    {
        // cout << "----------------------------" << std::endl;
        // cout << "Timestep " << timestep << std::endl;

        // find a plan
        sync_shared_env();
        // vector<Action> actions = planner->plan(plan_time_limit);
        // vector<Action> actions;
        // planner->plan(plan_time_limit,actions);

        auto start = std::chrono::steady_clock::now();

        vector<Action> actions = plan();

        ONLYDEV(
            if (actions.size()==num_of_agents) {
                analyzer.data["moving_steps"]=analyzer.data["moving_steps"].get<int>()+1;
            } else{
                if (actions.size()!=0) {
                    DEV_DEBUG("planner return wrong number of actions: {}", actions.size());
                    exit(-1);
                } else {
                    // DEV_DEBUG(fmt::format(fmt::fg(fmt::terminal_color::yellow )|fmt::emphasis::bold,"planner return no actions: most likely exceeding the time limit."));
                    DEV_WARN("planner return no actions: most likely exceeding the time limit.");
                }
            }
        )

        auto end = std::chrono::steady_clock::now();

        timestep += 1;
        ONLYDEV(analyzer.data["timesteps"]=timestep;)

        for (int a = 0; a < num_of_agents; a++)
        {
            if (!env->goal_locations[a].empty())
                solution_costs[a]++;
        }

        // move drives
        list<Task> new_finished_tasks = move(actions);
        if (!planner_movements[0].empty() && planner_movements[0].back() == Action::NA)
        {
            planner_times.back()+=plan_time_limit;  //add planning time to last record
        }
        else
        {
            auto diff = end-start;
            planner_times.push_back(std::chrono::duration<double>(diff).count());
        }
        // cout << new_finished_tasks.size() << " tasks has been finished in this timestep" << std::endl;

        // update tasks
        for (auto task : new_finished_tasks)
        {
            // int id, loc, t;
            // std::tie(id, loc, t) = task;
            finished_tasks[task.agent_assigned].emplace_back(task);
            num_of_tasks++;
            num_of_task_finish++;
        }
        // cout << num_of_tasks << " tasks has been finished by far in total" << std::endl;

        ONLYDEV(analyzer.data["finished_tasks"]=num_of_tasks;)

        update_tasks();

        bool complete_all = false;
        for (auto & t: assigned_tasks)
        {
            if(t.empty()) 
            {
                complete_all = true;
            }
            else
            {
                complete_all = false;
                break;
            }
        }
        if (complete_all)
        {
            cout << std::endl << "All task finished!" << std::endl;
            break;
        }

        ONLYDEV(g_timer.print_all_d(););
    }
    ONLYDEV(g_timer.record_d("initialize_end","simulate_end","simulation");)

    ONLYDEV(g_timer.print_all_d();)

    cout << std::endl << "Done!" << std::endl;
    cout << num_of_tasks << " tasks has been finished by far in total" << std::endl;

    ONLYDEV(analyzer.dump();)
}


void BaseSystem::initialize()
{
    paths.resize(num_of_agents);
    events.resize(num_of_agents);
    env->num_of_agents = num_of_agents;
    env->rows = map.rows;
    env->cols = map.cols;
    env->map = map.map;
    finished_tasks.resize(num_of_agents);
    // bool succ = load_records(); // continue simulating from the records
    timestep = 0;
    curr_states = starts;
    assigned_tasks.resize(num_of_agents);

    //planner initilise before knowing the first goals
    auto planner_initialize_success= planner_initialize();
    
    log_preprocessing(planner_initialize_success);
    if (!planner_initialize_success)
        return;

    // initialize_goal_locations();
    update_tasks();

    sync_shared_env();

    
    this->execution_paths.resize(num_of_agents);
    this->planning_paths.resize(num_of_agents);

    actual_movements.resize(num_of_agents);
    planner_movements.resize(num_of_agents);
    solution_costs.resize(num_of_agents);
    for (int a = 0; a < num_of_agents; a++)
    {
        solution_costs[a] = 0;
    }
}

void BaseSystem::savePaths(const string &fileName, int option) const
{
    std::ofstream output;
    output.open(fileName, std::ios::out);
    for (int i = 0; i < num_of_agents; i++)
    {
        output << "Agent " << i << ": ";
        if (option == 0)
        {
            bool first = true;
            for (const auto t : actual_movements[i])
            {
                if (!first)
                {
                    output << ",";
                }
                else
                {
                    first = false;
                }
                output << t;
            }
        }
        else if (option == 1)
        {
            bool first = true;
            for (const auto t : planner_movements[i])
            {
                if (!first)
                {
                    output << ",";
                } 
                else 
                {
                    first = false;
                }
                output << t;
            }
        }
        output << endl;
    }
    output.close();
}

#ifdef MAP_OPT

nlohmann::json BaseSystem::analyzeResults()
{
    json js;
    // Save action model
    js["actionModel"] = "MAPF_T";

    std::string feasible = fast_mover_feasible ? "Yes" : "No";
    js["AllValid"] = feasible;

    js["teamSize"] = num_of_agents;

    // Save start locations[x,y,orientation]
    json start = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        json s = json::array();
        s.push_back(starts[i].location/map.cols);
        s.push_back(starts[i].location%map.cols);
        switch (starts[i].orientation)
        {
        case 0:
            s.push_back("E");
            break;
        case 1:
            s.push_back("S");
        case 2:
            s.push_back("W");
            break;
        case 3:
            s.push_back("N");
            break;
        }
        start.push_back(s);
    }
    js["start"] = start;

    js["numTaskFinished"] = num_of_task_finish;
    int sum_of_cost = 0;
    int makespan = 0;
    if (num_of_agents > 0)
    {
        sum_of_cost = solution_costs[0];
        makespan = solution_costs[0];
        for (int a = 1; a < num_of_agents; a++)
        {
            sum_of_cost += solution_costs[a];
            if (solution_costs[a] > makespan)
            {
                makespan = solution_costs[a];
            }
        }
    }
    js["sumOfCost"] = sum_of_cost;
    js["makespan"] = makespan;
  
    // Save actual paths
    json apaths = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        std::string path;
        bool first = true;
        for (const auto action : actual_movements[i])
        {
            if (!first)
            {
                path+= ",";
            }
            else
            {
                first = false;
            }

            if (action == Action::FW)
            {
                path+="F";
            }
            else if (action == Action::CR)
            {
                path+="R";
            } 
            else if (action == Action::CCR)
            {
                path+="C";
            }
            else if (action == Action::NA)
            {
                path+="T";
            }
            else
            {
                path+="W";
            }
        }
        apaths.push_back(path);
    }
    js["actualPaths"] = apaths;

    //planned paths
    json ppaths = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        std::string path;
        bool first = true;
        for (const auto action : planner_movements[i])
        {
            if (!first)
            {
                path+= ",";
            } 
            else 
            {
                first = false;
            }

            if (action == Action::FW)
            {
                path+="F";
            }
            else if (action == Action::CR)
            {
                path+="R";
            } 
            else if (action == Action::CCR)
            {
                path+="C";
            } 
            else if (action == Action::NA)
            {
                path+="T";
            }
            else
            {
                path+="W";
            }
        }  
        ppaths.push_back(path);
    }
    js["plannerPaths"] = ppaths;

    json planning_times = json::array();
    for (double time: planner_times)
        planning_times.push_back(time);
    js["plannerTimes"] = planning_times;

    // Save errors
    json errors = json::array();
    for (auto error: model->errors)
    {
        std::string error_msg;
        int agent1;
        int agent2;
        int timestep;
        std::tie(error_msg,agent1,agent2,timestep) = error;
        json e = json::array();
        e.push_back(agent1);
        e.push_back(agent2);
        e.push_back(timestep);
        e.push_back(error_msg);
        errors.push_back(e);

    }
    js["errors"] = errors;
  
    // Save events
    json events_json = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        json event = json::array();
        for(auto e: events[i])
        {
            json ev = json::array();
            std::string event_msg;
            int task_id;
            int timestep;
            std::tie(task_id,timestep,event_msg) = e;
            ev.push_back(task_id);
            ev.push_back(timestep);
            ev.push_back(event_msg);
            event.push_back(ev);
        }
        events_json.push_back(event);
    }
    js["events"] = events_json;

    // Save all tasks
    json tasks = json::array();
    for (auto t: all_tasks)
    {
        json task = json::array();
        task.push_back(t.task_id);
        task.push_back(t.location/map.cols);
        task.push_back(t.location%map.cols);
        tasks.push_back(task);
    }
    js["tasks"] = tasks;

    json final_states = json::array();
    for (auto s: this->curr_states){
        final_states.push_back(s);
    }
    js["final_pos"] = final_states;

    json final_tasks = json::array();
    for (auto assigned_task_per_agent: this->assigned_tasks){
        final_tasks.push_back(assigned_task_per_agent.back());
    }
    js["final_tasks"] = final_tasks;

    json exec_p = json::array();
    for (int i=0; i<num_of_agents; ++i){
        json ss = json::array();
        for (const auto state: execution_paths[i]){
            json s = json::array();
            s.push_back(state.location/map.cols);
            s.push_back(state.location%map.cols);
            ss.push_back(s);
        }
        exec_p.push_back(ss);
    }
    js["execFuture"] = exec_p;

    json plan_p = json::array();
    for (int i=0; i<num_of_agents; ++i){
        json ss = json::array();
        for (const auto state: planning_paths[i]){
            json s = json::array();
            s.push_back(state.location/map.cols);
            s.push_back(state.location%map.cols);
            ss.push_back(s);
        }
        plan_p.push_back(ss);
    }
    js["planFuture"] = plan_p;

    return analyze_result_json(js, map);
}

#endif

void BaseSystem::saveResults(const string &fileName) const
{
    json js;
    // Save action model
    js["actionModel"] = "MAPF_T";

    std::string feasible = fast_mover_feasible ? "Yes" : "No";
    js["AllValid"] = feasible;

    js["teamSize"] = num_of_agents;

    // Save start locations[x,y,orientation]
    json start = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        json s = json::array();
        s.push_back(starts[i].location/map.cols);
        s.push_back(starts[i].location%map.cols);
        switch (starts[i].orientation)
        {
        case 0:
            s.push_back("E");
            break;
        case 1:
            s.push_back("S");
        case 2:
            s.push_back("W");
            break;
        case 3:
            s.push_back("N");
            break;
        }
        start.push_back(s);
    }
    js["start"] = start;

    js["numTaskFinished"] = num_of_task_finish;
    int sum_of_cost = 0;
    int makespan = 0;
    if (num_of_agents > 0)
    {
        sum_of_cost = solution_costs[0];
        makespan = solution_costs[0];
        for (int a = 1; a < num_of_agents; a++)
        {
            sum_of_cost += solution_costs[a];
            if (solution_costs[a] > makespan)
            {
                makespan = solution_costs[a];
            }
        }
    }
    js["sumOfCost"] = sum_of_cost;
    js["makespan"] = makespan;
  
    // Save actual paths
    json apaths = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        std::string path;
        bool first = true;
        for (const auto action : actual_movements[i])
        {
            if (!first)
            {
                path+= ",";
            }
            else
            {
                first = false;
            }

            if (action == Action::FW)
            {
                path+="F";
            }
            else if (action == Action::CR)
            {
                path+="R";
            } 
            else if (action == Action::CCR)
            {
                path+="C";
            }
            else if (action == Action::NA)
            {
                path+="T";
            }
            else
            {
                path+="W";
            }
        }
        apaths.push_back(path);
    }
    js["actualPaths"] = apaths;

    //planned paths
    json ppaths = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        std::string path;
        bool first = true;
        for (const auto action : planner_movements[i])
        {
            if (!first)
            {
                path+= ",";
            } 
            else 
            {
                first = false;
            }

            if (action == Action::FW)
            {
                path+="F";
            }
            else if (action == Action::CR)
            {
                path+="R";
            } 
            else if (action == Action::CCR)
            {
                path+="C";
            } 
            else if (action == Action::NA)
            {
                path+="T";
            }
            else
            {
                path+="W";
            }
        }  
        ppaths.push_back(path);
    }
    js["plannerPaths"] = ppaths;

    json planning_times = json::array();
    for (double time: planner_times)
        planning_times.push_back(time);
    js["plannerTimes"] = planning_times;

    // Save errors
    json errors = json::array();
    for (auto error: model->errors)
    {
        std::string error_msg;
        int agent1;
        int agent2;
        int timestep;
        std::tie(error_msg,agent1,agent2,timestep) = error;
        json e = json::array();
        e.push_back(agent1);
        e.push_back(agent2);
        e.push_back(timestep);
        e.push_back(error_msg);
        errors.push_back(e);

    }
    js["errors"] = errors;
  
    // Save events
    json events_json = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        json event = json::array();
        for(auto e: events[i])
        {
            json ev = json::array();
            std::string event_msg;
            int task_id;
            int timestep;
            std::tie(task_id,timestep,event_msg) = e;
            ev.push_back(task_id);
            ev.push_back(timestep);
            ev.push_back(event_msg);
            event.push_back(ev);
        }
        events_json.push_back(event);
    }
    js["events"] = events_json;

    // Save all tasks
    json tasks = json::array();
    for (auto t: all_tasks)
    {
        json task = json::array();
        task.push_back(t.task_id);
        task.push_back(t.location/map.cols);
        task.push_back(t.location%map.cols);
        tasks.push_back(task);
    }
    js["tasks"] = tasks;

    std::ofstream f(fileName,std::ios_base::trunc |std::ios_base::out);
    f << std::setw(4) << js;

}

bool FixedAssignSystem::load_agent_tasks(string fname)
{
    string line;
    std::ifstream myfile(fname.c_str());
    if (!myfile.is_open()) return false;

    getline(myfile, line);
    while (!myfile.eof() && line[0] == '#') {
        getline(myfile, line);
    }

    boost::char_separator<char> sep(",");
    boost::tokenizer<boost::char_separator<char>> tok(line, sep);
    boost::tokenizer<boost::char_separator<char>>::iterator beg = tok.begin();

    num_of_agents = atoi((*beg).c_str());
    int task_id = 0;
    // My benchmark
    if (num_of_agents == 0) {
        //issue_logs.push_back("Load file failed");
        std::cerr << "The number of agents should be larger than 0" << endl;
        exit(-1);
    }
    starts.resize(num_of_agents);
    task_queue.resize(num_of_agents);
  
    for (int i = 0; i < num_of_agents; i++)
    {
        cout << "agent " << i << ": ";

        getline(myfile, line);
        while (!myfile.eof() && line[0] == '#')
            getline(myfile, line);

        boost::tokenizer<boost::char_separator<char>> tok(line, sep);
        boost::tokenizer<boost::char_separator<char>>::iterator beg = tok.begin();
        // read start [row,col] for agent i
        int num_landmarks = atoi((*beg).c_str());
        beg++;
        auto loc = atoi((*beg).c_str());
        // agent_start_locations[i] = {loc, 0};
        starts[i] = State(loc, 0, 0);
        cout << loc;
        beg++;
        for (int j = 0; j < num_landmarks; j++, beg++)
        {
            auto loc = atoi((*beg).c_str());
            task_queue[i].emplace_back(task_id++, loc, 0, i);
            cout << " -> " << loc;
        }
        cout << endl;
    }
    myfile.close();

    return true;
}


void FixedAssignSystem::update_tasks()
{
    for (int k = 0; k < num_of_agents; k++)
    {
        while (assigned_tasks[k].size() < num_tasks_reveal && !task_queue[k].empty())
        {
            Task task = task_queue[k].front();
            task_queue[k].pop_front();
            assigned_tasks[k].push_back(task);
            events[k].push_back(make_tuple(task.task_id,timestep,"assigned"));
            all_tasks.push_back(task);
            log_event_assigned(k, task.task_id, timestep);
        }
    }
}


void TaskAssignSystem::update_tasks()
{
    for (int k = 0; k < num_of_agents; k++)
    {
        while (assigned_tasks[k].size() < num_tasks_reveal && !task_queue.empty())
        {
            std::cout << "assigned task " << task_queue.front().task_id <<
                " with loc " << task_queue.front().location << " to agent " << k << std::endl;
            Task task = task_queue.front();
            task.t_assigned = timestep;
            task.agent_assigned = k;
            task_queue.pop_front();
            assigned_tasks[k].push_back(task);
            events[k].push_back(make_tuple(task.task_id,timestep,"assigned"));
            all_tasks.push_back(task);
            log_event_assigned(k, task.task_id, timestep);
        }
    }
}


void InfAssignSystem::update_tasks(){
    for (int k = 0; k < num_of_agents; k++)
    {
        while (assigned_tasks[k].size() < num_tasks_reveal) 
        {
            int loc;
            if(this->init_task && this->init_task_ids[k]!=-1){
                loc = this->init_task_ids[k];
                this->init_task_ids[k] = -1;
            } else{
                int i = task_counter[k] * num_of_agents + k;
                loc = tasks[i%tasks_size];
            }
            Task task(task_id,loc,timestep,k);
            assigned_tasks[k].push_back(task);
            events[k].push_back(make_tuple(task.task_id,timestep,"assigned"));
            log_event_assigned(k, task.task_id, timestep);
            all_tasks.push_back(task);
            task_id++;
            task_counter[k]++;
        }
    }
}

void InfAssignSystem::resume_from_file(string snapshot_fp, int w){
    std::ifstream fin(snapshot_fp);
    int n;
    fin>>n;
    if (n!=num_of_agents) {
        std::cerr<<"number of agents in snapshot file does not match: "<<n<<" vs "<<num_of_agents<<std::endl;
    }
    for (int k = 0; k < num_of_agents; k++)
    {
        int x,y,o;
        fin>>x>>y>>o;
        int p=y*w+x;
        std::cout<<p<<" "<<x<<" "<<y<<" "<<o<<std::endl;
        starts[k]=State(p,0,o);
    }
}

#else

list<Task> BaseSystem::move(vector<Action>& actions)
{
    // actions.resize(num_of_agents, Action::NA);
    for (int k = 0; k < num_of_agents; k++)
    {
        //log->log_plan(false,k);
        if (k >= actions.size()){
            fast_mover_feasible = false;
            planner_movements[k].push_back(Action::NA);
        }
        else
        {
            planner_movements[k].push_back(actions[k]);
        }
    }

    list<Task> finished_tasks_this_timestep; // <agent_id, task_id, timestep>
    if (!valid_moves(curr_states, actions))
    {
        fast_mover_feasible = false;
        actions = std::vector<Action>(num_of_agents, Action::W);
    }

    curr_states = model->result_states(curr_states, actions);
    // agents do not move
    for (int k = 0; k < num_of_agents; k++)
    {
        if (!assigned_tasks[k].empty() && curr_states[k].location == assigned_tasks[k].front().location)
        {
            Task task = assigned_tasks[k].front();
            assigned_tasks[k].pop_front();
            task.t_completed = timestep;
            finished_tasks_this_timestep.push_back(task);
            events[k].push_back(make_tuple(task.task_id, timestep,"finished"));
            log_event_finished(k, task.task_id, timestep);
        }
        paths[k].push_back(curr_states[k]);
        actual_movements[k].push_back(actions[k]);
    }

    return finished_tasks_this_timestep;
}


// This function might not work correctly with small map (w or h <=2)
bool BaseSystem::valid_moves(vector<State>& prev, vector<Action>& action)
{
    return model->is_valid(prev, action);
}


void BaseSystem::sync_shared_env() {
    env->goal_locations.resize(num_of_agents);
    for (size_t i = 0; i < num_of_agents; i++)
    {
        env->goal_locations[i].clear();
        for (auto& task: assigned_tasks[i])
        {
            env->goal_locations[i].push_back({task.location, task.t_assigned });
        }
    }
    env->curr_timestep = timestep;
    env->curr_states = curr_states;
}


vector<Action> BaseSystem::plan_wrapper()
{
    ONLYDEV(std::cout<<"wrapper called"<<std::endl;)
    vector<Action> actions;
    vector<list<State>> cur_exec_paths(num_of_agents);
    vector<list<State>> cur_plan_paths(num_of_agents);
    ONLYDEV(std::cout<<"planning"<<std::endl;)
    planner->plan(plan_time_limit, actions, cur_exec_paths, cur_plan_paths);

    this->execution_paths = cur_exec_paths;
    this->planning_paths = cur_plan_paths;
    // cout << "exec:";
    // for (int i=0; i<num_of_agents; ++i){
    //     for (const auto loc: cur_exec_paths[i]){
    //         cout << loc.location << " ";
    //     }
    // }
    // cout << endl;
    // cout << "plan:";
    // for (int i=0; i<num_of_agents; ++i){
    //     for (const auto loc: cur_plan_paths[i]){
    //         cout << loc.location;
    //     }
    // }
    // cout << endl;

    return actions;
}


vector<Action> BaseSystem::plan()
{
    return plan_wrapper();

    // using namespace std::placeholders;
    // if (started && future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    // {
    //     std::cout << started << "     " << (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) << std::endl;
    //     if(logger)
    //     {
    //         logger->log_info("planner cannot run because the previous run is still running", timestep);
    //     }

    //     if (future.wait_for(std::chrono::seconds(plan_time_limit)) == std::future_status::ready)
    //     {
    //         task_td.join();
    //         started = false;
    //         return future.get();
    //     }
    //     logger->log_info("planner timeout", timestep);
    //     return {};
    // }

    // std::packaged_task<std::vector<Action>()> task(std::bind(&BaseSystem::plan_wrapper, this));
    // future = task.get_future();
    // if (task_td.joinable())
    // {
    //     task_td.join();
    // }
    // task_td = std::thread(std::move(task));
    // started = true;
    // if (future.wait_for(std::chrono::seconds(plan_time_limit)) == std::future_status::ready)
    // {
    //     task_td.join();
    //     started = false;
    //     return future.get();
    // }
    // logger->log_info("planner timeout", timestep);
    // return {};
}


bool BaseSystem::planner_initialize()
{
    planner->initialize(preprocess_time_limit);
    return true;

    // using namespace std::placeholders;
    // std::packaged_task<void(int)> init_task(std::bind(&MAPFPlanner::initialize, planner, std::placeholders::_1));
    // auto init_future = init_task.get_future();

    // auto init_td = std::thread(std::move(init_task), preprocess_time_limit);
    // if (init_future.wait_for(std::chrono::seconds(preprocess_time_limit)) == std::future_status::ready)
    // {
    //     init_td.join();
    //     return true;
    // }

    // init_td.detach();
    // return false;
}


void BaseSystem::log_preprocessing(bool succ)
{
    if (logger == nullptr)
        return;
    if (succ)
    {
        logger->log_info("Preprocessing success", timestep);
    } 
    else
    {
        logger->log_fatal("Preprocessing timeout", timestep);
    }
}


void BaseSystem::log_event_assigned(int agent_id, int task_id, int timestep)
{
    ONLYDEV(logger->log_info("Task " + std::to_string(task_id) + " is assigned to agent " + std::to_string(agent_id), timestep);)
}


void BaseSystem::log_event_finished(int agent_id, int task_id, int timestep) 
{
    ONLYDEV(logger->log_info("Agent " + std::to_string(agent_id) + " finishes task " + std::to_string(task_id), timestep);)
}

void BaseSystem::warmup(int total_warmup_steps){
    initialize();
    for (; this->warmupstep< total_warmup_steps;){
        sync_shared_env();
        auto start = std::chrono::steady_clock::now();
        vector<Action> actions = plan();
        auto end = std::chrono::steady_clock::now();
        this->warmupstep+=1;

        for (int a = 0; a < num_of_agents; a++){
            if (!env->goal_locations[a].empty())
                solution_costs[a]++;
        }

        // move drives
        list<Task> new_finished_tasks = move(actions);
        if (!planner_movements[0].empty() && planner_movements[0].back() == Action::NA){
            planner_times.back()+=plan_time_limit;  //add planning time to last record
        } else{
            auto diff = end-start;
            planner_times.push_back(std::chrono::duration<double>(diff).count());
        }

        // TODO: ensure warmup steps does not contribute to the whole setting
        this->curr_finish_task_agents.clear();
        for (auto task : new_finished_tasks){
            finished_tasks[task.agent_assigned].emplace_back(task);
            this->curr_finish_task_agents.push_back(task.agent_assigned);
            // num_of_task_finish++;
        }
        update_tasks();

        bool complete_all = false;
        for (auto & t: assigned_tasks){
            if(t.empty()){
                complete_all = true;
            }else{
                complete_all = false;
                break;
            }
        }
        if (complete_all){
            cout << std::endl << "All task finished!" << std::endl;
            break;
        }
    }
}

int BaseSystem::update_gg_and_step(int update_gg_interval){
    
    this->curr_starts = this->curr_states;

    int t_step=0;
    for (; t_step<update_gg_interval && this->timestep < this->total_simulation_steps;){
        sync_shared_env();
        auto start = std::chrono::steady_clock::now();
        vector<Action> actions = plan();
        auto end = std::chrono::steady_clock::now();
        t_step += 1;
        timestep += 1;

        for (int a = 0; a < num_of_agents; a++){
            if (!env->goal_locations[a].empty())
                solution_costs[a]++;
        }

        // move drives
        list<Task> new_finished_tasks = move(actions);
        if (!planner_movements[0].empty() && planner_movements[0].back() == Action::NA){
            planner_times.back()+=plan_time_limit;  //add planning time to last record
        } else{
            auto diff = end-start;
            planner_times.push_back(std::chrono::duration<double>(diff).count());
        }

        this->curr_finish_task_agents.clear();
        for (auto task : new_finished_tasks){
            finished_tasks[task.agent_assigned].emplace_back(task);
            this->curr_finish_task_agents.push_back(task.agent_assigned);
            num_of_task_finish++;
        }

        if (this->task_dist_change_interval>0 && this->timestep%this->task_dist_change_interval == 0){
            this->random_update_tasks_distribution();
        }

        update_tasks();

        bool complete_all = false;
        for (auto & t: assigned_tasks){
            if(t.empty()){
                complete_all = true;
            }else{
                complete_all = false;
                break;
            }
        }
        if (complete_all){
            cout << std::endl << "All task finished!" << std::endl;
            break;
        }
    }
    return t_step;
    // exit(1);
}

void BaseSystem::simulate(int simulation_time)
{
    //init logger
    //Logger* log = new Logger();

    ONLYDEV(g_timer.record_p("simulate_start");)

    initialize();

    ONLYDEV(g_timer.record_d("simulate_start","initialize_end","initialization");)
    int num_of_tasks = 0;

    for (; timestep < simulation_time; )
    {
        ONLYDEV(
            cout << "----------------------------" << std::endl;
            cout << "Timestep " << timestep << std::endl;
        )

        // find a plan
        sync_shared_env();
        // vector<Action> actions = planner->plan(plan_time_limit);
        // vector<Action> actions;
        // planner->plan(plan_time_limit,actions);

        auto start = std::chrono::steady_clock::now();

        vector<Action> actions = plan();

        ONLYDEV(
            if (actions.size()==num_of_agents) {
                analyzer.data["moving_steps"]=analyzer.data["moving_steps"].get<int>()+1;
            } else{
                if (actions.size()!=0) {
                    DEV_DEBUG("planner return wrong number of actions: {}", actions.size());
                    exit(-1);
                } else {
                    DEV_WARN("planner return no actions: most likely exceeding the time limit.");
                }
            }
        )

        auto end = std::chrono::steady_clock::now();

        timestep += 1;
        ONLYDEV(analyzer.data["timesteps"]=timestep;)

        for (int a = 0; a < num_of_agents; a++)
        {
            if (!env->goal_locations[a].empty())
                solution_costs[a]++;
        }

        // move drives
        list<Task> new_finished_tasks = move(actions);
        if (!planner_movements[0].empty() && planner_movements[0].back() == Action::NA)
        {
            planner_times.back()+=plan_time_limit;  //add planning time to last record
        }
        else
        {
            auto diff = end-start;
            planner_times.push_back(std::chrono::duration<double>(diff).count());
        }
        ONLYDEV(cout << new_finished_tasks.size() << " tasks has been finished in this timestep" << std::endl;)

        // update tasks
        this->curr_finish_task_agents.clear();
        for (auto task : new_finished_tasks)
        {
            // int id, loc, t;
            // std::tie(id, loc, t) = task;
            finished_tasks[task.agent_assigned].emplace_back(task);
            this->curr_finish_task_agents.push_back(task.agent_assigned);

            num_of_tasks++;
            num_of_task_finish++;
        }
        ONLYDEV(cout << num_of_tasks << " tasks has been finished by far in total" << std::endl;)

        ONLYDEV(analyzer.data["finished_tasks"]=num_of_tasks;)

        if (this->task_dist_change_interval>0 && this->timestep%this->task_dist_change_interval == 0){
            this->random_update_tasks_distribution();
        }

        update_tasks();

        bool complete_all = false;
        for (auto & t: assigned_tasks)
        {
            if(t.empty()) 
            {
                complete_all = true;
            }
            else
            {
                complete_all = false;
                break;
            }
        }
        if (complete_all)
        {
            cout << std::endl << "All task finished!" << std::endl;
            break;
        }

        ONLYDEV(g_timer.print_all_d(););
    }
    ONLYDEV(g_timer.record_d("initialize_end","simulate_end","simulation");)

    ONLYDEV(g_timer.print_all_d();)

    cout << std::endl << "Done!" << std::endl;
    cout << num_of_tasks << " tasks has been finished by far in total" << std::endl;

    ONLYDEV(analyzer.dump();)
}


void BaseSystem::initialize()
{
    paths.resize(num_of_agents);
    events.resize(num_of_agents);
    env->num_of_agents = num_of_agents;
    env->rows = map.rows;
    env->cols = map.cols;
    env->map = map.map;
    finished_tasks.resize(num_of_agents);
    // bool succ = load_records(); // continue simulating from the records
    timestep = 0;
    this->warmupstep = 0;
    curr_states = starts;
    this->curr_starts = this->curr_states;
    assigned_tasks.resize(num_of_agents);

    //planner initilise before knowing the first goals
    auto planner_initialize_success= planner_initialize();
    
    log_preprocessing(planner_initialize_success);
    if (!planner_initialize_success)
        return;

    // initialize_goal_locations();
    update_tasks();

    sync_shared_env();

    this->execution_paths.resize(num_of_agents);
    this->planning_paths.resize(num_of_agents);

    actual_movements.resize(num_of_agents);
    planner_movements.resize(num_of_agents);
    solution_costs.resize(num_of_agents);
    for (int a = 0; a < num_of_agents; a++)
    {
        solution_costs[a] = 0;
    }
}

void BaseSystem::savePaths(const string &fileName, int option) const
{
    std::ofstream output;
    output.open(fileName, std::ios::out);
    for (int i = 0; i < num_of_agents; i++)
    {
        output << "Agent " << i << ": ";
        if (option == 0)
        {
            bool first = true;
            for (const auto t : actual_movements[i])
            {
                if (!first)
                {
                    output << ",";
                }
                else
                {
                    first = false;
                }
                output << t;
            }
        }
        else if (option == 1)
        {
            bool first = true;
            for (const auto t : planner_movements[i])
            {
                if (!first)
                {
                    output << ",";
                } 
                else 
                {
                    first = false;
                }
                output << t;
            }
        }
        output << endl;
    }
    output.close();
}

#ifdef MAP_OPT

nlohmann::json BaseSystem::analyzeCurrResults(int update_gg_interval)
{
    json js;
    // Save action model
    js["actionModel"] = "MAPF";

    std::string feasible = fast_mover_feasible ? "Yes" : "No";
    js["AllValid"] = feasible;

    js["teamSize"] = num_of_agents;

    // Save start locations[x,y,orientation]
    json curr_start = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        json s = json::array();
        s.push_back(curr_starts[i].location/map.cols);
        s.push_back(curr_starts[i].location%map.cols);
        // s.push_back(starts[i].location/map.cols);
        // s.push_back(starts[i].location%map.cols);
        curr_start.push_back(s);
        // std::cout << "agent"<<i<<", curr states ="<<this->curr_states[i].location/map.cols<<", "<<this->curr_states[i].location%map.cols<<std::endl;
    }
    js["start"] = curr_start;

    js["numTaskFinishedSoFar"] = num_of_task_finish;

    // Save actual paths
    json apaths = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        std::string path;
        bool first = true;

        if (update_gg_interval > actual_movements[i].size()){
            std::cout << "must have bug here!" <<std::endl;
        }
        auto it = std::prev(actual_movements[i].end(), update_gg_interval);
        // auto it = actual_movements[i].begin();
        // for (const auto action : actual_movements[i])
        for (; it!=actual_movements[i].end(); ++it){
            auto action = *it;
            if (!first){
                path+= ",";
            } else{
                first = false;
            }

            if (action == Action::R){
                path+="R";
            } else if (action == Action::D){
                path+="D";
            } else if (action == Action::L){
                path+="L";
            } else if (action == Action::U){
                path+="U";
            } else if (action == Action::W){
                path+="W";
            } else{
                path+="X";
            }
        }
        apaths.push_back(path);
    }
    js["actualPaths"] = apaths;

    json final_states = json::array();
    for (auto s: this->curr_states){
        final_states.push_back(s.location);
    }
    js["final_pos"] = final_states;

    json final_tasks = json::array();
    for (auto assigned_task_per_agent: this->assigned_tasks){
        final_tasks.push_back(assigned_task_per_agent.back().location);
    }
    js["final_tasks"] = final_tasks;

    json exec_p = json::array();
    for (int i=0; i<num_of_agents; ++i){
        json ss = json::array();
        for (const auto state: execution_paths[i]){
            json s = json::array();
            s.push_back(state.location/map.cols);
            s.push_back(state.location%map.cols);
            ss.push_back(s);
        }
        exec_p.push_back(ss);
    }
    js["execFuture"] = exec_p;

    json plan_p = json::array();
    for (int i=0; i<num_of_agents; ++i){
        json ss = json::array();
        for (const auto state: planning_paths[i]){
            json s = json::array();
            s.push_back(state.location/map.cols);
            s.push_back(state.location%map.cols);
            ss.push_back(s);
        }
        plan_p.push_back(ss);
    }
    js["planFuture"] = plan_p;

    json agents_finish_task = json::array();
    for (auto a_id: this->curr_finish_task_agents){
        agents_finish_task.push_back(a_id);
    }
    js["agents_finish_task"] = agents_finish_task;

    js["done"] = (this->timestep >= this->total_simulation_steps);

    return analyze_curr_result_json(js, map);
}

nlohmann::json BaseSystem::analyzeResults()
{
    json js;
    // Save action model
    js["actionModel"] = "MAPF";

    std::string feasible = fast_mover_feasible ? "Yes" : "No";
    js["AllValid"] = feasible;

    js["teamSize"] = num_of_agents;

    // Save start locations[x,y,orientation]
    json start = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        json s = json::array();
        s.push_back(starts[i].location/map.cols);
        s.push_back(starts[i].location%map.cols);
        switch (starts[i].orientation)
        {
        case 0:
            s.push_back("E");
            break;
        case 1:
            s.push_back("S");
        case 2:
            s.push_back("W");
            break;
        case 3:
            s.push_back("N");
            break;
        }
        start.push_back(s);
    }
    js["start"] = start;

    js["numTaskFinished"] = num_of_task_finish;
    int sum_of_cost = 0;
    int makespan = 0;
    if (num_of_agents > 0)
    {
        sum_of_cost = solution_costs[0];
        makespan = solution_costs[0];
        for (int a = 1; a < num_of_agents; a++)
        {
            sum_of_cost += solution_costs[a];
            if (solution_costs[a] > makespan)
            {
                makespan = solution_costs[a];
            }
        }
    }
    js["sumOfCost"] = sum_of_cost;
    js["makespan"] = makespan;
  
    // Save actual paths
    json apaths = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        std::string path;
        bool first = true;
        for (const auto action : actual_movements[i])
        {
            if (!first)
            {
                path+= ",";
            }
            else
            {
                first = false;
            }

            if (action == Action::R)
            {
                path+="R";
            }
            else if (action == Action::D)
            {
                path+="D";
            } 
            else if (action == Action::L)
            {
                path+="L";
            }
            else if (action == Action::U)
            {
                path+="U";
            }
            else if (action == Action::W)
            {
                path+="W";
            }
            else
            {
                path+="X";
            }
        }
        apaths.push_back(path);
    }
    js["actualPaths"] = apaths;

    //planned paths
    json ppaths = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        std::string path;
        bool first = true;
        for (const auto action : planner_movements[i])
        {
            if (!first)
            {
                path+= ",";
            } 
            else 
            {
                first = false;
            }

            if (action == Action::R)
            {
                path+="R";
            }
            else if (action == Action::D)
            {
                path+="D";
            } 
            else if (action == Action::L)
            {
                path+="L";
            }
            else if (action == Action::U)
            {
                path+="U";
            }
            else if (action == Action::W)
            {
                path+="W";
            }
            else
            {
                path+="X";
            }
        }  
        ppaths.push_back(path);
    }
    js["plannerPaths"] = ppaths;

    json planning_times = json::array();
    for (double time: planner_times)
        planning_times.push_back(time);
    js["plannerTimes"] = planning_times;

    // Save errors
    json errors = json::array();
    for (auto error: model->errors)
    {
        std::string error_msg;
        int agent1;
        int agent2;
        int timestep;
        std::tie(error_msg,agent1,agent2,timestep) = error;
        json e = json::array();
        e.push_back(agent1);
        e.push_back(agent2);
        e.push_back(timestep);
        e.push_back(error_msg);
        errors.push_back(e);

    }
    js["errors"] = errors;
  
    // Save events
    json events_json = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        json event = json::array();
        for(auto e: events[i])
        {
            json ev = json::array();
            std::string event_msg;
            int task_id;
            int timestep;
            std::tie(task_id,timestep,event_msg) = e;
            ev.push_back(task_id);
            ev.push_back(timestep);
            ev.push_back(event_msg);
            event.push_back(ev);
        }
        events_json.push_back(event);
    }
    js["events"] = events_json;

    // Save all tasks
    json tasks = json::array();
    for (auto t: all_tasks)
    {
        json task = json::array();
        task.push_back(t.task_id);
        task.push_back(t.location/map.cols);
        task.push_back(t.location%map.cols);
        tasks.push_back(task);
    }
    js["tasks"] = tasks;

    json final_states = json::array();
    for (auto s: this->curr_states){
        final_states.push_back(s.location);
    }
    js["final_pos"] = final_states;

    json final_tasks = json::array();
    for (auto assigned_task_per_agent: this->assigned_tasks){
        final_tasks.push_back(assigned_task_per_agent.back().location);
    }
    js["final_tasks"] = final_tasks;

    json agents_finish_task = json::array();
    for (auto a_id: this->curr_finish_task_agents){
        agents_finish_task.push_back(a_id);
    }
    js["agents_finish_task"] = agents_finish_task;

    json exec_p = json::array();
    for (int i=0; i<num_of_agents; ++i){
        json ss = json::array();
        for (const auto state: execution_paths[i]){
            json s = json::array();
            s.push_back(state.location/map.cols);
            s.push_back(state.location%map.cols);
            ss.push_back(s);
        }
        exec_p.push_back(ss);
    }
    js["execFuture"] = exec_p;

    json plan_p = json::array();
    for (int i=0; i<num_of_agents; ++i){
        json ss = json::array();
        for (const auto state: planning_paths[i]){
            json s = json::array();
            s.push_back(state.location/map.cols);
            s.push_back(state.location%map.cols);
            ss.push_back(s);
        }
        plan_p.push_back(ss);
    }
    js["planFuture"] = plan_p;

    js["done"] = (this->timestep >= this->total_simulation_steps);

    return analyze_result_json(js, map);
}

#endif

void BaseSystem::saveResults(const string &fileName) const
{
    json js;
    // Save action model
    js["actionModel"] = "MAPF";

    std::string feasible = fast_mover_feasible ? "Yes" : "No";
    js["AllValid"] = feasible;

    js["teamSize"] = num_of_agents;

    // Save start locations[x,y,orientation]
    json start = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        json s = json::array();
        s.push_back(starts[i].location/map.cols);
        s.push_back(starts[i].location%map.cols);
        switch (starts[i].orientation)
        {
        case 0:
            s.push_back("E");
            break;
        case 1:
            s.push_back("S");
        case 2:
            s.push_back("W");
            break;
        case 3:
            s.push_back("N");
            break;
        }
        start.push_back(s);
    }
    js["start"] = start;

    js["numTaskFinished"] = num_of_task_finish;
    int sum_of_cost = 0;
    int makespan = 0;
    if (num_of_agents > 0)
    {
        sum_of_cost = solution_costs[0];
        makespan = solution_costs[0];
        for (int a = 1; a < num_of_agents; a++)
        {
            sum_of_cost += solution_costs[a];
            if (solution_costs[a] > makespan)
            {
                makespan = solution_costs[a];
            }
        }
    }
    js["sumOfCost"] = sum_of_cost;
    js["makespan"] = makespan;
  
    // Save actual paths
    json apaths = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        std::string path;
        bool first = true;
        for (const auto action : actual_movements[i])
        {
            if (!first)
            {
                path+= ",";
            }
            else
            {
                first = false;
            }

            if (action == Action::R)
            {
                path+="R";
            }
            else if (action == Action::D)
            {
                path+="D";
            } 
            else if (action == Action::L)
            {
                path+="L";
            }
            else if (action == Action::U)
            {
                path+="U";
            }
            else if (action == Action::W)
            {
                path+="W";
            }
            else
            {
                path+="X";
            }
        }
        apaths.push_back(path);
    }
    js["actualPaths"] = apaths;

    //planned paths
    json ppaths = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        std::string path;
        bool first = true;
        for (const auto action : planner_movements[i])
        {
            if (!first)
            {
                path+= ",";
            } 
            else 
            {
                first = false;
            }

            if (action == Action::R)
            {
                path+="R";
            }
            else if (action == Action::D)
            {
                path+="D";
            } 
            else if (action == Action::L)
            {
                path+="L";
            }
            else if (action == Action::U)
            {
                path+="U";
            }
            else if (action == Action::W)
            {
                path+="W";
            }
            else
            {
                path+="X";
            }
        }  
        ppaths.push_back(path);
    }
    js["plannerPaths"] = ppaths;

    json planning_times = json::array();
    for (double time: planner_times)
        planning_times.push_back(time);
    js["plannerTimes"] = planning_times;

    // Save errors
    json errors = json::array();
    for (auto error: model->errors)
    {
        std::string error_msg;
        int agent1;
        int agent2;
        int timestep;
        std::tie(error_msg,agent1,agent2,timestep) = error;
        json e = json::array();
        e.push_back(agent1);
        e.push_back(agent2);
        e.push_back(timestep);
        e.push_back(error_msg);
        errors.push_back(e);

    }
    js["errors"] = errors;
  
    // Save events
    json events_json = json::array();
    for (int i = 0; i < num_of_agents; i++)
    {
        json event = json::array();
        for(auto e: events[i])
        {
            json ev = json::array();
            std::string event_msg;
            int task_id;
            int timestep;
            std::tie(task_id,timestep,event_msg) = e;
            ev.push_back(task_id);
            ev.push_back(timestep);
            ev.push_back(event_msg);
            event.push_back(ev);
        }
        events_json.push_back(event);
    }
    js["events"] = events_json;

    // Save all tasks
    json tasks = json::array();
    for (auto t: all_tasks)
    {
        json task = json::array();
        task.push_back(t.task_id);
        task.push_back(t.location/map.cols);
        task.push_back(t.location%map.cols);
        tasks.push_back(task);
    }
    js["tasks"] = tasks;

    std::ofstream f(fileName,std::ios_base::trunc |std::ios_base::out);
    f << std::setw(4) << js;

}

bool FixedAssignSystem::load_agent_tasks(string fname)
{
    string line;
    std::ifstream myfile(fname.c_str());
    if (!myfile.is_open()) return false;

    getline(myfile, line);
    while (!myfile.eof() && line[0] == '#') {
        getline(myfile, line);
    }

    boost::char_separator<char> sep(",");
    boost::tokenizer<boost::char_separator<char>> tok(line, sep);
    boost::tokenizer<boost::char_separator<char>>::iterator beg = tok.begin();

    num_of_agents = atoi((*beg).c_str());
    int task_id = 0;
    // My benchmark
    if (num_of_agents == 0) {
        //issue_logs.push_back("Load file failed");
        std::cerr << "The number of agents should be larger than 0" << endl;
        exit(-1);
    }
    starts.resize(num_of_agents);
    task_queue.resize(num_of_agents);
  
    for (int i = 0; i < num_of_agents; i++)
    {
        cout << "agent " << i << ": ";

        getline(myfile, line);
        while (!myfile.eof() && line[0] == '#')
            getline(myfile, line);

        boost::tokenizer<boost::char_separator<char>> tok(line, sep);
        boost::tokenizer<boost::char_separator<char>>::iterator beg = tok.begin();
        // read start [row,col] for agent i
        int num_landmarks = atoi((*beg).c_str());
        beg++;
        auto loc = atoi((*beg).c_str());
        // agent_start_locations[i] = {loc, 0};
        starts[i] = State(loc, 0, 0);
        cout << loc;
        beg++;
        for (int j = 0; j < num_landmarks; j++, beg++)
        {
            auto loc = atoi((*beg).c_str());
            task_queue[i].emplace_back(task_id++, loc, 0, i);
            cout << " -> " << loc;
        }
        cout << endl;
    }
    myfile.close();

    return true;
}


void FixedAssignSystem::update_tasks()
{
    for (int k = 0; k < num_of_agents; k++)
    {
        while (assigned_tasks[k].size() < num_tasks_reveal && !task_queue[k].empty())
        {
            Task task = task_queue[k].front();
            task_queue[k].pop_front();
            assigned_tasks[k].push_back(task);
            events[k].push_back(make_tuple(task.task_id,timestep,"assigned"));
            all_tasks.push_back(task);
            log_event_assigned(k, task.task_id, timestep);
        }
    }
}


void TaskAssignSystem::update_tasks()
{
    for (int k = 0; k < num_of_agents; k++)
    {
        while (assigned_tasks[k].size() < num_tasks_reveal && !task_queue.empty())
        {
            std::cout << "assigned task " << task_queue.front().task_id <<
                " with loc " << task_queue.front().location << " to agent " << k << std::endl;
            Task task = task_queue.front();
            task.t_assigned = timestep;
            task.agent_assigned = k;
            task_queue.pop_front();
            assigned_tasks[k].push_back(task);
            events[k].push_back(make_tuple(task.task_id,timestep,"assigned"));
            all_tasks.push_back(task);
            log_event_assigned(k, task.task_id, timestep);
        }
    }
}


void InfAssignSystem::update_tasks(){
    for (int k = 0; k < num_of_agents; k++)
    {
        while (assigned_tasks[k].size() < num_tasks_reveal) 
        {
            int loc;
            if(this->init_task && this->init_task_ids[k]!=-1){
                loc = this->init_task_ids[k];
                this->init_task_ids[k] = -1;
            } else{
                int i = task_counter[k] * num_of_agents + k;
                loc = tasks[i%tasks_size];
            }

            Task task(task_id,loc,timestep,k);
            assigned_tasks[k].push_back(task);
            events[k].push_back(make_tuple(task.task_id,timestep,"assigned"));
            log_event_assigned(k, task.task_id, timestep);
            all_tasks.push_back(task);
            task_id++;
            task_counter[k]++;
        }
    }
}

void KivaSystem::update_tasks_base_distribution(std::vector<double> new_distribution){
    int home_loc_size = this->home_loc_weights.size();
    int end_pts_size = this->end_pts_weights.size();
    if (new_distribution.size()!= home_loc_size+end_pts_size){
        std::cout << "Error for distribution size! "<<std::endl;
        std::cout << "home_loc_size ="<<home_loc_size<<", end_pts_size ="<<end_pts_size<<", ";
        std::cout << "but get input size ="<<new_distribution.size()<<std::endl;
        exit(1);
    }
    for(int i=0; i<home_loc_size; ++i){
        this->home_loc_weights[i] = new_distribution[i];
    }
    for(int i=0; i<end_pts_size; ++i){
        this->end_pts_weights[i] = new_distribution[home_loc_size+i];
    }
    this->update_tasks_distribution();
}

void KivaSystem::update_tasks_distribution(){
    this->agent_home_loc_dist = std::discrete_distribution<int>(
        this->home_loc_weights.begin(),
        this->home_loc_weights.end()
    );
    
    this->agent_end_pts_dist = std::discrete_distribution<int>(
        this->end_pts_weights.begin(),
        this->end_pts_weights.end()
    );
}

std::vector<double> KivaSystem::get_tasks_distribution(){
    std::vector<double> dist;
    for (auto w: this->home_loc_weights){
        dist.push_back(w);
    }
    for (auto w: this->end_pts_weights){
        dist.push_back(w);
    }
    return dist;
}


void KivaSystem::update_tasks(){
    for (int k = 0; k < num_of_agents; k++)
    {
        while (assigned_tasks[k].size() < num_tasks_reveal) 
        {
            int prev_task_loc=prev_task_locs[k];

            int loc;
            if(this->init_task && this->init_task_ids[k]!=-1){
                loc = this->init_task_ids[k];
                this->init_task_ids[k] = -1;
            } else{
                if (map.grid_types[prev_task_loc]=='.' || map.grid_types[prev_task_loc]=='e') 
                {
                    // next task would be w
                    // Sample a workstation based on given distribution
                    int idx = this->agent_home_loc_dist(this->MT);
                    // int idx=MT()%map.agent_home_locations.size();
                    loc=map.agent_home_locations[idx];
                } else if (map.grid_types[prev_task_loc]=='w')
                {
                    // next task would e
                    // int idx=MT()%map.end_points.size();
                    int idx = this->agent_end_pts_dist(this->MT);
                    loc=map.end_points[idx];
                } else {
                    std::cout<<"unknown grid type"<<std::endl;
                    exit(-1);
                }
            }
            
            Task task(task_id,loc,timestep,k);
            assigned_tasks[k].push_back(task);
            events[k].push_back(make_tuple(task.task_id,timestep,"assigned"));
            log_event_assigned(k, task.task_id, timestep);
            all_tasks.push_back(task);
            prev_task_locs[k]=loc;
            task_id++;
        }
    }
}

void KivaSystem::random_update_tasks_distribution(){
    // std::cout << "begin distribution update"<<std::endl;
    if (this->random_type == "Gaussian"){
        generateTaskAndAgentGaussianDist(this->map, this->MT, this->home_loc_weights, this->end_pts_weights);
    } else if (this->random_type == "GaussianMixed"){
        generateTaskAndAgentMultiModeGaussianDist(this->map, this->MT, this->home_loc_weights, this->end_pts_weights);
    } else if (this->random_type == "GaussianMixedRandomK"){
        generateTaskAndAgentMultiModeGaussianRandomKDist(this->map, this->MT, this->home_loc_weights, this->end_pts_weights);
    } else if (this->random_type == "LR"){
        generateTaskAndAgentLRDist(this->map, this->MT, this->home_loc_weights, this->end_pts_weights);
        
        // debug
        std::cout << "home loc:" <<std::endl;
        for (auto w: this->home_loc_weights){
            std::cout << w <<", ";
        }
        std::cout << std::endl;

        // std::cout << "end loc" <<std::endl;
        // for (auto w: this->end_pts_weights){
        //     std::cout << w <<", ";
        // }
        // std::cout << std::endl;
    } else {
        std::cout << "random type ["<< this->random_type<< "] not support yet"<<std::endl;
        exit(1);
    }
    
    this->update_tasks_distribution();
    // std::cout << "end distribution update"<<std::endl;
}


void InfAssignSystem::resume_from_file(string snapshot_fp, int w){
    std::ifstream fin(snapshot_fp);
    int n;
    fin>>n;
    if (n!=num_of_agents) {
        std::cerr<<"number of agents in snapshot file does not match: "<<n<<" vs "<<num_of_agents<<std::endl;
    }
    for (int k = 0; k < num_of_agents; k++)
    {
        int x,y;
        fin>>x>>y;
        int p=y*w+x;
        std::cout<<p<<" "<<x<<" "<<y<<std::endl;
        starts[k]=State(p,0,-1);
    }
}

void OnlineGenerateTaskSystem::update_tasks(){
    for (int k=0; k<this->num_of_agents; ++k){
        while(assigned_tasks[k].size() < this->num_tasks_reveal){
            int loc;
            if(this->init_task && this->init_task_ids[k]!=-1){
                loc = this->init_task_ids[k];
                this->init_task_ids[k] = -1;
            } else {
                int idx = this->goal_loc_dist(this->MT);
                loc = this->map.empty_locations[idx];
            }

            Task task(this->task_id, loc, this->timestep, k);
            assigned_tasks[k].push_back(task);
            events[k].push_back(make_tuple(task.task_id, this->timestep, "assigned"));
            log_event_assigned(k, task.task_id, timestep);
            all_tasks.push_back(task);
            task_id++;
        }
    }
}

void OnlineGenerateTaskSystem::random_update_tasks_distribution(){
    // std::cout << "calling Online::random_update_task_distribution, random_type =" <<this->random_type <<std::endl;
    if (this->random_type == "uniform"){
        this->empty_weights.resize(this->map.empty_locations.size(), 1);
    } else if (this->random_type == "Gaussian"){
        generateTaskAndAgentGaussianEmptyDist(this->map, this->MT, this->empty_weights);
    } else if (this->random_type == "GaussianMixed"){
        generateMultiModeGaussianEmptyDist(this->map, this->MT, this->empty_weights);
    } else {
        std::cout << "random type [" << this->random_type << "] is not supported in OnlineGenerateSystem yet!" <<std::endl;
        exit(1);
    }
    this->goal_loc_dist = std::discrete_distribution<int>(
        this->empty_weights.begin(), this->empty_weights.end()
    );
}

void MultiCategoryTaskSystem::random_update_tasks_distribution(){
    double p1 = 0.1/7;
    double p2 = 0.2/2;
    double p3 = 0.7/1;
    std::uniform_real_distribution<> dis(0.0, 1.0);

    this->empty_weights.clear();
    for(int i=0; i<this->map.empty_locations.size(); ++i){
        double q = dis(this->MT);
        if (q<0.7){
            this->empty_weights.push_back(p1);
        } else if (q<0.9){
            this->empty_weights.push_back(p2);
        } else {
            this->empty_weights.push_back(p3);
        }
    }
    
    this->goal_loc_dist = std::discrete_distribution<int>(
        this->empty_weights.begin(), this->empty_weights.end()
    );
}

void MultiCategoryTaskSystem::update_tasks(){
    for (int k=0; k<this->num_of_agents; ++k){
        while(assigned_tasks[k].size() < this->num_tasks_reveal){
            int loc;
            if(this->init_task && this->init_task_ids[k]!=-1){
                loc = this->init_task_ids[k];
                this->init_task_ids[k] = -1;
            } else {
                int idx = this->goal_loc_dist(this->MT);
                loc = this->map.empty_locations[idx];
            }

            Task task(this->task_id, loc, this->timestep, k);
            assigned_tasks[k].push_back(task);
            events[k].push_back(make_tuple(task.task_id, this->timestep, "assigned"));
            log_event_assigned(k, task.task_id, timestep);
            all_tasks.push_back(task);
            task_id++;
        }
    }
}

std::vector<double> MultiCategoryTaskSystem::get_tasks_distribution(){
    return this->empty_weights;
}

std::vector<double> OnlineGenerateTaskSystem::get_tasks_distribution(){
    return this->empty_weights;
}
#endif