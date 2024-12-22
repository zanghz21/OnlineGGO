

#include "flow.hpp"
#include "dev.h"

#include <random>
#include <unordered_set>

namespace TrafficMAPF{

// std::random_device rd;
std::mt19937 g(0);

int get_all_flow_for_nn(const int flow_sum){
    int all_flow = flow_sum;
    if (!network_config.use_all_flow){
        return 10000;
    }
    if (all_flow == 0) {
        all_flow ++;
    }
    return all_flow;
}

int get_op_flow(TrajLNS& lns, int prev_loc, int loc, int d){
    int op_flow = lns.flow[prev_loc].d[d] * lns.flow[loc].d[(d+2)%4];// / (lns.flow[prev_loc].d[d] + lns.flow[loc].d[(d+2)%4]+1);
    return op_flow;
}

int get_vertex_flow(TrajLNS& lns, int loc){
    int vertex_flow = 0 ;
    for (int k = 0; k < 4; k++){
        vertex_flow += lns.flow[loc].d[k];
    }
    return (vertex_flow -1) /2;
}

int get_all_op_flow(TrajLNS& lns){
    int op_flow = 0;

    for (int row=0; row < lns.env->rows; row++){
        for (int col=0; col < lns.env->cols; col++){
            int loc = row*lns.env->cols + col;
            int right_loc = loc + 1;
            int bottom_loc = loc + lns.env->cols;
            
            if (col != lns.env->cols-1)  
                op_flow += lns.flow[loc].d[0] * lns.flow[right_loc].d[2];
            if (row != lns.env->rows-1)
                op_flow += lns.flow[loc].d[1] * lns.flow[bottom_loc].d[3];

        }
    }
    
    return op_flow;
}


double get_nn_flow(
    TrajLNS& lns, int prev_loc, int loc, int d, int all_flow, std::shared_ptr<Network> network_ptr 
){
    if (network_config.has_path || network_config.has_previous){
        std::cout << "NOT support previous/path observation in LNS yet" <<std::endl;
        exit(1);
    }
    std::vector<double> flow_input;
    std::vector<double> map_input;
    std::vector<int> ids;

    if (network_config.input_type == "flow"){
        get_flow_NN_input(lns.env, lns.flow, all_flow, prev_loc, flow_input, map_input, ids);
    } else if (network_config.input_type == "traf_and_goal"){
        get_traf_and_goal_NN_input(lns.env, prev_loc, flow_input, map_input, ids);
    } else if (network_config.input_type == "flow_and_traf_and_goal"){
        get_flow_and_traf_and_goal_NN_input(lns.env, lns.flow, all_flow, prev_loc, flow_input, map_input, ids);
    } else {
        std::cout << "error net input_type!" <<std::endl;
        exit(1);
    }
    
    // std::cout << "after get input" <<std::endl;
    int output_dim = 0;
    if (network_config.output_size == 4){
        output_dim = d;
    }   
    
    double raw_cost;
    raw_cost = network_ptr->forward(output_dim, flow_input, map_input);
    return raw_cost;
}

//remove flow for each location's outgoing edge according to the traj
void remove_traj(TrajLNS& lns, int agent){
    lns.occupations[lns.trajs[agent].front()].erase(agent);
    lns.soc -= lns.trajs[agent].size() - 1;
    if (lns.trajs[agent].size() <= 1){
        return;
    }
    int loc, prev_loc, diff, d;
    int old_op_flow, old_vertex_flow, new_op_flow, new_vertex_flow;
    int diff_op_flow, diff_vertex_flow;
    for (int j = 1; j < lns.trajs[agent].size(); j++){
        loc = lns.trajs[agent][j];
        prev_loc = lns.trajs[agent][j-1];
        diff = loc - prev_loc;
        d = get_d(diff, lns.env);

        lns.occupations[loc].erase(agent);

        old_op_flow = get_op_flow(lns, prev_loc, loc, d);
        old_vertex_flow = get_vertex_flow(lns, prev_loc);
        assert(lns.op_flow == get_all_op_flow(lns));


        //update the new flow;
        lns.flow[prev_loc].d[d] -= 1;
        lns.flow_sum -= 1;
        if (OBJECTIVE == OBJ::NN){
            if (network_config.input_type == "flow"
            || network_config.input_type == "flow_and_traf_and_goal"){
                lns.set_nn_cached_invalid(prev_loc);
            }
        }

        new_op_flow = get_op_flow(lns, prev_loc, loc, d);
        new_vertex_flow = get_vertex_flow(lns, prev_loc);


        //calculate the different on op_flow and vertex flow
        //update lns.op_flow and lns.vertex_flow
        diff_op_flow = new_op_flow - old_op_flow;
        diff_vertex_flow = new_vertex_flow - old_vertex_flow;

        lns.op_flow += diff_op_flow;
        lns.vertex_flow += diff_vertex_flow;
        assert(lns.flow[prev_loc].d[d] >= 0);
        assert(lns.op_flow >= 0);
        assert(lns.vertex_flow >= 0);
        assert(lns.op_flow == get_all_op_flow(lns));
    }
    lns.nn_flow -= lns.goal_nodes[agent].nn_cost;
    // if (lns.nn_flow < 0){
    //     std::cout << "lns nn flow should not less than 0! currently ="<< lns.nn_flow <<std::endl;
    //     exit(-1);
    // }
}

void add_traj(TrajLNS& lns, int agent, std::shared_ptr<Network> network_ptr){
    lns.occupations[lns.trajs[agent].front()].insert(agent);

    lns.soc += lns.trajs[agent].size() - 1;
    if (lns.trajs[agent].size() <= 1){
        return;
    }
    int loc, prev_loc, diff, d;
    int old_op_flow, old_vertex_flow, new_op_flow, new_vertex_flow;
    int nn_all_flow;
    double old_nn_flow, new_nn_flow;
    int diff_op_flow, diff_vertex_flow;
    double diff_nn_flow;

    for (int j = 1; j < lns.trajs[agent].size(); j++){
        loc = lns.trajs[agent][j];
        prev_loc = lns.trajs[agent][j-1];
        diff = loc - prev_loc;
        d = get_d(diff, lns.env);

        lns.occupations[loc].insert(agent);


        old_op_flow = get_op_flow(lns, prev_loc, loc, d);
        old_vertex_flow = get_vertex_flow(lns, prev_loc);
        
        // #ifdef GUIDANCE_LNS
        // if (network_ptr != nullptr){
        //     nn_all_flow = get_all_flow_for_nn(lns.flow);
        //     old_nn_flow = get_nn_flow(lns, prev_loc, loc, d, nn_all_flow, network_ptr);
        // }
        // #endif
        // assert(lns.op_flow == get_all_op_flow(lns));
        // std::std::cout<<"old:"<<lns.op_flow<<","<<lns.flow[prev_loc].d[d]<<","<<lns.flow[loc].d[(d+2)%4]<<std::endl;

        
        //update the new flow;
        lns.flow[prev_loc].d[d] += 1;
        lns.flow_sum +=1;
        if (OBJECTIVE == OBJ::NN){
            if (network_config.input_type == "flow"
            || network_config.input_type == "flow_and_traf_and_goal"){
                lns.set_nn_cached_invalid(prev_loc);
            }
        }
        // std::cout << "flow add at [" << prev_loc<<"], d=" <<d <<", flow ="<<lns.flow[prev_loc].d[d]<<std::endl;

        new_op_flow = get_op_flow(lns, prev_loc, loc, d);
        new_vertex_flow = get_vertex_flow(lns, prev_loc);

        // #ifdef GUIDANCE_LNS
        // if (network_ptr != nullptr){
        //     nn_all_flow += 1;
        //     new_nn_flow = get_nn_flow(lns, prev_loc, loc, d, nn_all_flow, network_ptr);
        // }
        // #endif

        //calculate the different on op_flow and vertex flow
        //update lns.op_flow and lns.vertex_flow
        diff_op_flow = new_op_flow - old_op_flow;
        diff_vertex_flow = new_vertex_flow - old_vertex_flow;

        // #ifdef GUIDANCE_LNS
        // if (network_ptr != nullptr){
        //     diff_nn_flow = new_nn_flow - old_nn_flow;
        // }
        // #endif

        lns.op_flow += diff_op_flow;
        lns.vertex_flow += diff_vertex_flow;

        // #ifdef GUIDANCE_LNS
        // if (network_ptr != nullptr){
        //     lns.nn_flow += diff_nn_flow;
        // }
        // #endif
        // std::std::cout<<"new:"<<lns.op_flow<<","<<lns.flow[prev_loc].d[d]<<","<<lns.flow[loc].d[(d+2)%4]<<std::endl;

        assert(lns.op_flow >= 0);
        assert(lns.vertex_flow >= 0);
        assert(lns.op_flow == get_all_op_flow(lns));

    }
    lns.nn_flow += lns.goal_nodes[agent].nn_cost;
}

void init_traj(TrajLNS& lns,std::vector<int>& traffic,int amount, std::shared_ptr<Network> network_ptr, bool remove_and_plan){
    //trajs is a vector of Traj, each Traj is a vector of int
    //trajs[i] is a vector of int, each int is a location
    //trajs[i][j] is a location


    //compute the traj for each agent using aStarOF
    //update the flow according to the traj
    int count = 0 ;
    for(int i=0;i<lns.env->num_of_agents;i++){
        if (lns.t_ms !=0 && std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - lns.start_time).count() >lns.t_ms){
            break;
        }
        // std::cout<<i<<";";

        if (count >= amount){
            break;
        }
        if (remove_and_plan &&  !lns.trajs[i].empty()){
            remove_traj(lns,i);
        }

        if (lns.trajs[i].empty() || remove_and_plan){
            int start = lns.env->curr_states[i].location;
            int goal = lns.tasks[i];
            lns.goal_nodes[i] = aStarOF(lns.env,lns.flow, lns.flow_sum, lns.heuristics[goal],traffic,lns.trajs[i],lns.mem,start,goal, network_ptr, lns.nn_cached_value);

            add_traj(lns,i, network_ptr);
            count++;
            lns.traj_inited++;
        }

    }
    ONLYDEV(std::cout<<"---init-agents:"<<count<<std::endl;)
    

}

void init_traj_st(TrajLNS& lns,std::vector<int>& traffic){
    //trajs is a vector of Traj, each Traj is a vector of int
    //trajs[i] is a vector of int, each int is a location
    //trajs[i][j] is a location


    //compute the traj for each agent using aStarOF
    //update the flow according to the traj
    int count = 0 ;
    for(int i=0;i<lns.env->num_of_agents;i++){
        // if (count >= amount){
        //     break;
        // }
        if (lns.trajs[i].empty()){
            int start = lns.env->curr_states[i].location;
            int goal = lns.tasks[i];

            lns.goal_nodes[i] = singleShortestPath(lns.env,lns.flow, lns.heuristics[goal],traffic,lns.trajs[i],lns.mem,start,goal);

            count++;
            lns.traj_inited++;
        }

    }

    for(int i=0;i<lns.env->num_of_agents;i++){
        // if (count >= amount){
        //     break;
        // }
        if(!lns.trajs[i].empty()){
            add_traj(lns,i);
        }

    }
    

}

void update_dist_2_path(TrajLNS& lns, int i, std::vector<int>& traffic){
    init_dist_2_path(lns.traj_dists[i], lns.env, lns.trajs[i], traffic);
}

//compute distance table for each traj
void init_dist_table(TrajLNS& lns,std::vector<int>& traffic, int amount){

    int count = 0;
    for (int i=0 ; i <lns.env->num_of_agents; i++){
                // std::cout<<i<<";";
        if (count >= amount){
            break;
        }
        if(!lns.trajs[i].empty() && lns.trajs[i].size() == get_heuristic(lns.heuristics[lns.trajs[i].back()], lns.env, traffic, lns.flow,lns.trajs[i].front()))
            continue;
        if(!lns.trajs[i].empty() && lns.traj_dists[i].empty()){
            init_dist_2_path(lns.traj_dists[i], lns.env, lns.trajs[i], traffic);
            count++;
            lns.dist2path_inited++;
        }

    }
}

//update traj and distance table for agent i
void update_traj(TrajLNS& lns, int i, std::vector<int>& traffic, std::shared_ptr<Network> network_ptr){
    int start = lns.env->curr_states[i].location;
    int goal = lns.tasks[i];
    lns.goal_nodes[i] = aStarOF(lns.env,lns.flow, lns.flow_sum, lns.heuristics[goal],traffic,lns.trajs[i],lns.mem,start,goal, network_ptr, lns.nn_cached_value);
    add_traj(lns,i, network_ptr);
}




bool terminate(TrajLNS& lns,int max_ite, double time_limit, int ite, TimePoint start_time, int stop){
    if (ite >= max_ite){
        return true;
    }

    //time_limit in seconds
    if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() >= time_limit*1000){
        return true;
    }
    if (stop){
        return true;
    }


    return false;

}

void random_select(TrajLNS& lns, std::vector<int> & agents){
    //randomly select const lns.group_size agents 
    //and store the index of the agents in agents
    if (lns.group_size >= lns.env->num_of_agents){
        for (int i = 0; i < lns.env->num_of_agents; i++){
            agents[i] = i;
        }
        return;
    }
    int num_of_agents = lns.env->num_of_agents;
    int i = 0;
    while ( i < lns.group_size){
        int index = rand() % num_of_agents;
        //check if index is already in agents
        bool in = false;
        for (int j = 0; j < i; j++){
            if (agents[j] == index){
                in = true;
                break;
            }
        }
        if (in){
            continue;
        }
        agents[i] = index;
        i += 1;
    }
}

void select_most_expensive(TrajLNS& lns, std::vector<int> & agents){
    if (lns.num_in_tablu >= lns.env->num_of_agents/2){
        lns.tabu_list.clear();
        lns.tabu_list.resize(lns.env->num_of_agents,false);
        lns.num_in_tablu = 0;
    }
    int most_op_flow_i = -1, most_cost_i= -1;
    int most_cost = 0, most_op_flow =0;
    for (int i = 0;i < lns.env->num_of_agents; i++){
        if (lns.tabu_list[i]){
            continue;
        }
        if (lns.goal_nodes[i].get_op_flow() > most_op_flow){
            most_op_flow = lns.goal_nodes[i].get_op_flow();
            most_op_flow_i = i;
        }
        if (lns.goal_nodes[i].get_g() > most_cost){
            most_cost = lns.goal_nodes[i].get_g();
            most_cost_i = i;
        }
    }

    int agent = most_op_flow_i == -1? most_cost_i : most_op_flow_i;
    std::vector<int> congest_agents;

    if (agent != -1){
        lns.tabu_list[agent] = true;
        lns.num_in_tablu += 1;

        Traj& path = lns.trajs[agent];
        for (int i = 0; i < path.size(); i++){
            int loc = path[i];
            if (lns.occupations[loc].size() > loc){
                congest_agents.clear();
                congest_agents.insert(congest_agents.begin(),lns.occupations[loc].begin(),lns.occupations[loc].end());
            }
        }
    }

    int i = 0;
    while ( i < lns.group_size){
        if (i < congest_agents.size()){
            agents[i] = congest_agents[i];
            i++;
            continue;
        }

        int index = rand() % lns.env->num_of_agents;
        //check if index is already in agents
        bool in = false;
        for (int j = 0; j < i; j++){
            if (agents[j] == index){
                in = true;
                break;
            }
        }
        if (in){
            continue;
        }
        agents[i] = index;
        i += 1;
    }
}


int select_method(TrajLNS& lns){
    double sum = 0;
    for (const auto& h : lns.weights)
        sum += h;
    double r = (double) rand() / RAND_MAX;
    double threshold = lns.weights[0];
    int method = 0;
    while (threshold < r * sum)
    {
        method++;
        threshold += lns.weights[method];
    }
    return method;
}


//neighborhood search
void destory_improve(TrajLNS& lns, std::vector<int>& traffic,std::unordered_set<int>& updated, int max_ite, double time_limit, std::shared_ptr<Network> network_ptr){
    int ite = 0;
    TimePoint start_time = std::chrono::steady_clock::now();
    bool stop = false;
    std::vector<int> neighbors(lns.group_size);
    std::vector<Traj> traj_copy;
    std::vector<s_node> goal_nodes_copy;

    TimePoint start = std::chrono::steady_clock::now();
    int t;
    
    int old_op_flow, old_vertex_flow, old_soc, method;
    double old_nn_flow;
    while(!terminate(lns, max_ite,time_limit,ite,start_time,stop)){
        if (lns.t_ms !=0 && std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - lns.start_time).count() >lns.t_ms){
            break;
        }

        //reset neighbors to -1
        for (int i = 0; i < lns.group_size; i++){
            neighbors[i] = -1;
        }

#ifndef LNS_DES_METHOD
        method = select_method(lns);
#else   
        method = LNS_DES_METHOD;
#endif

        //randomly select lns.group_size agents
        switch (method)
        {
        case ADAPTIVE::RANDOM:
            random_select(lns, neighbors);
            break;
        case ADAPTIVE::CONGESTION:
            select_most_expensive(lns, neighbors);
            break;
        default:
            std::cout<<"wrong method"<<std::endl;
            exit(1);
            break;
        }
        //shuffle the order of the agents
        std::shuffle(neighbors.begin(), neighbors.end(),g);


        //for each agent, remove its traj and update the flow
        old_op_flow = lns.op_flow;
        old_vertex_flow = lns.vertex_flow;
        old_soc = lns.soc;
        old_nn_flow = lns.nn_flow;
        traj_copy.clear();
        goal_nodes_copy.clear();
        for (int i = 0; i < lns.group_size; i++){
            int agent = neighbors[i];
            traj_copy.push_back(lns.trajs[agent]);
            goal_nodes_copy.push_back(lns.goal_nodes[agent]);
            remove_traj(lns, agent);
        }



        //for each agent, update its traj and update the flow
        for (int i = 0; i < lns.group_size; i++){
            int agent = neighbors[i];
            update_traj(lns, agent, traffic, network_ptr);
        }




        bool adopt_new;
        //if the new flow is not better, revert the change
        if ( 
            (OBJECTIVE == OBJ::O_VC && (lns.op_flow > old_op_flow || (lns.op_flow == old_op_flow &&  lns.vertex_flow+lns.soc > old_vertex_flow+old_soc)))
            ||
            (OBJECTIVE == OBJ::VC && (lns.vertex_flow+lns.soc > old_vertex_flow+old_soc))
            ||
            (OBJECTIVE == OBJ::SUM_OVC && (lns.op_flow + lns.vertex_flow+lns.soc > old_op_flow + old_vertex_flow+old_soc))
            ||
            (OBJECTIVE == OBJ::NN && (lns.soc+lns.nn_flow > old_soc+old_nn_flow))
            ||
            (OBJECTIVE == OBJ::OFFLINE && (lns.soc > old_soc))
            // ||
            // (OBJECTIVE == OBJ::NN && (lns.op_flow + lns.vertex_flow+lns.soc > old_op_flow + old_vertex_flow+old_soc))
            )
        {
            
            for (int i = 0; i < lns.group_size; i++){
                int agent = neighbors[i];
                remove_traj(lns, agent);
                lns.trajs[agent] = traj_copy[i];
                lns.goal_nodes[agent] = goal_nodes_copy[i];
                add_traj(lns, agent, network_ptr);
            }

            lns.weights[method] =
                        (1 - lns.decay_factor) * lns.weights[method];

            adopt_new = false;

        }
        else{
            // assert(lns.op_flow < old_op_flow || ( lns.op_flow == old_op_flow && lns.vertex_flow <= old_vertex_flow));
            for (int i = 0; i < lns.group_size; i++){
                int agent = neighbors[i];
                updated.insert(agent);
            }


            double improvement = 0.0;
            if (OBJECTIVE == OBJ::NN){
                improvement = (old_nn_flow+old_soc) - (lns.nn_flow+lns.soc);
            } else if (OBJECTIVE == OBJ::OFFLINE){
                improvement = old_soc - lns.soc;
            } else {
                improvement = std::max(old_op_flow - lns.op_flow, 0);
                if (lns.vertex_flow+lns.soc <  old_vertex_flow+old_soc){
                    improvement += old_vertex_flow + old_soc - lns.vertex_flow - lns.soc;
                }
            }
            


            lns.weights[method] =
                        lns.reaction_factor * (improvement)/lns.group_size
                         + (1-lns.reaction_factor)*lns.weights[method];

            adopt_new = true;
        }
                
#ifdef LNS_EXPOUT
        t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count(); 
        ONLYDEV(std::cout<<"---lns-ite,"<<ite<<","<< adopt_new <<","<< method<<","<<lns.weights[0] <<","<<lns.weights[1]<<","<<
            "initial,"<<old_op_flow<<","<<old_vertex_flow<<","<<old_soc<<","
            <<"after,"<<lns.op_flow<<","<<lns.vertex_flow<<","<<lns.soc<<","<< t <<std::endl;)
#endif



        ite += 1;
    }
}

}
