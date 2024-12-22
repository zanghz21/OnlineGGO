


#include "search.hpp"

namespace TrafficMAPF{
std::chrono::nanoseconds t;
//a astar minimized the opposide traffic flow with existing traffic flow

void get_extra_path_prev_obs(
    const std::vector<int> ids, 
    std::vector<double>& map_input, 
    s_node* curr, int center_id){
    if (!network_config.has_path && !network_config.has_previous){
        return;
    }
    std::unordered_map<int, int> id_hashmap;
    for (size_t i=0; i<ids.size();++i){
        id_hashmap[ids[i]]=i;
    }
    if (network_config.has_path){
        s_node* temp_ptr = curr;
        while(temp_ptr != nullptr){
            auto it = id_hashmap.find(temp_ptr->id);
            if (it != id_hashmap.end()){
                map_input[it->second] = -2.0;
            }
            temp_ptr = temp_ptr->parent;
        }
    }
    if (network_config.has_previous){
        if ((curr->id)==center_id){
            std::cout << "NOT support previous pos observation if centered at curr" << std::endl;
            exit(1);
        }
        auto it = id_hashmap.find(curr->id);
        if(it == id_hashmap.end()){
            std::cout << "curr search node must be included in the observation" <<std::endl;
            exit(1);
        }
        map_input[it->second] = -1.0;
    }
    return;
}

// window observation for guide-path edge usage
void get_flow_NN_input(SharedEnvironment* env, std::vector<Int4>& flow, int all_flow, 
    int center_id, std::vector<double>& flow_input, std::vector<double>& map_input, std::vector<int>& ids){
    
    flow_input.clear();
    map_input.clear();
    ids.clear();

    int WIN_R = network_config.WIN_R;
    for(int k1=-WIN_R; k1<=WIN_R; ++k1){
        for(int k2=-WIN_R; k2<=WIN_R; ++k2){
            int temp_id = center_id + k1*env->cols + k2;
            ids.push_back(temp_id);
            if (temp_id<0 || temp_id>=env->map.size()){
                for (int i=0; i<4; ++i){
                    flow_input.push_back(network_config.default_obst_flow);
                }
                map_input.push_back(-1.);
            } else {
                Int4 temp_flow = flow[temp_id];
                for (int i=0; i<4; ++i){
                    int next_id = get_id_from_d(i, temp_id, env);
                    if (next_id!=-1 && env->map[next_id]!=1){
                        flow_input.push_back((temp_flow.d[i] * 1.0)/((all_flow* 1.0) / 100.0));
                    } else {
                        flow_input.push_back(network_config.default_obst_flow);
                    }
                }
                map_input.push_back(double(env->map[temp_id]));
            }
        }
    }
}

void get_traf_and_goal_NN_input(SharedEnvironment* env, int center_id, 
    std::vector<double>& nn_input, std::vector<double>& map_input, std::vector<int>& ids){
    // traf+goal, id*6+[0, 4] -> traf, id*6+5 -> goal
    nn_input.clear();
    map_input.clear();
    ids.clear();

    int WIN_R = network_config.WIN_R;

    for(int k1=-WIN_R; k1<=WIN_R; ++k1){
        for(int k2=-WIN_R; k2<=WIN_R; ++k2){
            int temp_id = center_id + k1*env->cols + k2;
            ids.push_back(temp_id);
            if (temp_id<0 || temp_id>=env->map.size()){
                for (int i=0; i<5; ++i){
                    nn_input.push_back(network_config.default_obst_flow); //TODO: change the var name
                }
                nn_input.push_back(0.0); // goal observation
                map_input.push_back(-1.);
            } else {
                for (int i=0; i<5; ++i){
                    nn_input.push_back((env->hist_edge_usage[temp_id][i]*1.0)/env->past_traffic_interval);
                }
                nn_input.push_back((env->goal_loc_arr[temp_id] * 1.0)/env->num_of_agents); // goal_observation
                map_input.push_back(double(env->map[temp_id]));
            }
        }
    }
}

void get_flow_and_traf_and_goal_NN_input(SharedEnvironment* env, 
    std::vector<Int4>& flow, int all_flow, 
    int center_id, 
    std::vector<double>& nn_input, std::vector<double>& map_input, std::vector<int>& ids){
    // traf+goal, id*10+[0, 3] -> flow, id*10+[4, 8] -> traf, id*10+9 -> goal
    nn_input.clear();
    map_input.clear();
    ids.clear();

    int WIN_R = network_config.WIN_R;

    for(int k1=-WIN_R; k1<=WIN_R; ++k1){
        for(int k2=-WIN_R; k2<=WIN_R; ++k2){
            int temp_id = center_id + k1*env->cols + k2;
            // std::cout << "temp_id = "<<temp_id<<std::endl;
            ids.push_back(temp_id);
            if (temp_id<0 || temp_id>=env->map.size()){
                // guide-path edge usage
                for (int i=0; i<4; ++i){
                    nn_input.push_back(network_config.default_obst_flow);
                }
                // past edge usage
                for (int i=0; i<5; ++i){
                    nn_input.push_back(network_config.default_obst_flow);
                }
                nn_input.push_back(0.0); // goal observation
                map_input.push_back(-1.);
            } else {
                // guide-path edge usage
                Int4 temp_flow = flow[temp_id];
                for (int i=0; i<4; ++i){
                    int next_id = get_id_from_d(i, temp_id, env);
                    if (next_id!=-1 && env->map[next_id]!=1){
                        nn_input.push_back((temp_flow.d[i] * 1.0)/((all_flow* 1.0) / 100.0));
                    } else {
                        nn_input.push_back(network_config.default_obst_flow);
                    }
                }
                
                // past edge usage, include self-edges
                for (int i=0; i<5; ++i){
                    nn_input.push_back((env->hist_edge_usage[temp_id][i]*1.0)/env->past_traffic_interval);
                }

                // goal-loc
                nn_input.push_back((env->goal_loc_arr[temp_id] * 1.0)/env->num_of_agents);

                // map obs
                map_input.push_back(double(env->map[temp_id]));
            }
        }
    }
    // std::cout << "end" <<std::endl;
}

void get_minimum_NN_input(SharedEnvironment* env, std::vector<Int4>& flow, int all_flow, 
    int center_id, std::vector<double>& flow_input, std::vector<double>& map_input, std::vector<int>& ids){
    // NOTE: currently we do not use all flow to normalize
    // center-r, center-d, center-l, center-u, right-r, ..., down-r, ..., left-r, ..., up-r, ...
    flow_input.clear();
    map_input.clear();
    ids.clear();

    // center loc
    for (int i=0; i<4; ++i){
        flow_input.push_back(flow[center_id].d[i]);
        map_input.push_back(double(env->map[center_id]));
    }

    // TODO: omit redundant input
    int offsets[4] = {1, env->cols, -1, -env->cols};
    for (auto offset: offsets){
        // std::cout << "offset:" <<offset<<std::endl;
        int tmp_id = center_id + offset;
        if (tmp_id <0 || tmp_id >= env->map.size()){
            for (int i=0; i<4; ++i){
                flow_input.push_back(0.0);
            }
            map_input.push_back(-1.);
        } else {
            for (int i=0; i<4; ++i){
                flow_input.push_back(flow[tmp_id].d[i]);
            }
            map_input.push_back(double(env->map[tmp_id]));
        }
    }
}

// d = 0, 1, 2, 3 right, down, left, up (get_d(next-curr))
void get_directed_NN_input(SharedEnvironment* env, std::vector<Int4>& flow, int all_flow, 
    int center_id, std::vector<double>& flow_input, std::vector<double>& map_input, std::vector<int>& ids, int d){
    
    flow_input.clear();
    map_input.clear();
    ids.clear();

    int WIN_R = network_config.WIN_R;
    if (d == 0){
        for(int k2=WIN_R; k2>=-WIN_R; --k2){
            for(int k1=-WIN_R; k1<=WIN_R; ++k1){
                int temp_id = center_id + k1*env->cols + k2;
                ids.push_back(temp_id);
            }
        }
    } else if (d == 1){
        for(int k1=WIN_R; k1>=-WIN_R; --k1){
            for(int k2=WIN_R; k2>=-WIN_R; --k2){
                int temp_id = center_id + k1*env->cols + k2;
                ids.push_back(temp_id);
            }
        }
    } else if (d == 2){
        for(int k2=-WIN_R; k2<=WIN_R; ++k2){
            for(int k1=WIN_R; k1>=-WIN_R; --k1){
                int temp_id = center_id + k1*env->cols + k2;
                ids.push_back(temp_id);
            }
        }
    } else if (d == 3){
        for(int k1=-WIN_R; k1<=WIN_R; ++k1){
            for(int k2=-WIN_R; k2<=WIN_R; ++k2){
                int temp_id = center_id + k1*env->cols + k2;
                ids.push_back(temp_id);
            }
        }
    } else {
        std::cout << "direction d must in 0, 1, 2, 3, but receive "<< d <<std::endl;
        exit(-1);
    }
    for (auto temp_id: ids){
        if (temp_id<0 || temp_id>=env->map.size()){
            for (int i=0; i<4; ++i){
                flow_input.push_back(0.);
            }
            map_input.push_back(-1.);
        } else {
            Int4 temp_flow = flow[temp_id];
            for (int i=0; i<4; ++i){
                flow_input.push_back((temp_flow.d[i] * 1.0)/((all_flow* 1.0) / 100.0));
            }
            map_input.push_back(double(env->map[temp_id]));
        }
    }
}

// currently deprecate...
double get_next_NN_COST(
    SharedEnvironment* env, std::vector<Int4>& flow, std::shared_ptr<Network> network_ptr, int all_flow, 
    int next_id, s_node* curr
){
    std::vector<double> flow_input;
    std::vector<double> map_input;
    std::vector<int> ids;

    int center_id;
    get_flow_NN_input(env, flow, all_flow, next_id, flow_input, map_input, ids);
    get_extra_path_prev_obs(ids, map_input, curr, next_id);
    
    // std::cout << "after get input" <<std::endl;
    int d = get_d(next_id - curr->id,env);
    int output_dim = 0;
    if (network_config.output_size == 4){
        output_dim = (d+2)%4;
    }               
    
    double raw_cost;
    raw_cost = network_ptr->forward(output_dim, flow_input, map_input);
    return raw_cost;
}

std::vector<double> get_curr_NN_COST(
    SharedEnvironment* env, std::vector<Int4>& flow, std::shared_ptr<Network> network_ptr, int all_flow, 
    s_node* curr, std::vector<std::pair<bool, std::vector<double>>>& nn_cached_value
){
    // get weights from current location to all its neighbors
    if (network_config.use_cached_nn){
        if (nn_cached_value[curr->id].first){
            // std::cout << "hit" << std::endl;
            return nn_cached_value[curr->id].second;
        } else {
            // std::cout << "miss" <<std::endl;
            nn_cached_value[curr->id].first = true;
        }
    }

    std::vector<double> nn_input;
    std::vector<double> map_input;
    std::vector<int> ids;

    // Generate observation
    // auto t1 = std::chrono::high_resolution_clock::now();
    // std::cout << "network input_type ="<< network_config.input_type<<std::endl;
    if (network_config.input_type == "flow"){
        // By default, the obs is guide-path edge usage
        get_flow_NN_input(env, flow, all_flow, curr->id, nn_input, map_input, ids);
    } else if (network_config.input_type == "traf_and_goal"){
        // use (past edge usage, goal loc) as obs
        get_traf_and_goal_NN_input(env, curr->id, nn_input, map_input, ids);
    } else if (network_config.input_type == "flow_and_traf_and_goal") {
        // combination of the previous 2 types
        get_flow_and_traf_and_goal_NN_input(env, flow, all_flow, curr->id, nn_input, map_input, ids);
    } else if (network_config.input_type == "minimum"){
        // guidance policy with less parameters
        get_minimum_NN_input(env, flow, all_flow, curr->id, nn_input, map_input, ids);
    } else {
        std::cout << "error network input_type ["<<network_config.input_type<<"]!"<<std::endl;
        exit(-1);
    }
    // auto t2 = std::chrono::high_resolution_clock::now();
    
    std::vector<double> result;
    // auto t3 = std::chrono::high_resolution_clock::now();
    if (network_config.has_map){
        get_extra_path_prev_obs(ids, map_input, curr, curr->id);
    }

    // call network
    result = network_ptr->forward(nn_input, map_input);
    // auto t4 = std::chrono::high_resolution_clock::now();
    
    if (network_config.use_cached_nn){
        nn_cached_value[curr->id].second = result;
    }
    // std::cout << "get input time ="<<std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count()<<std::endl;
    // std::cout << "forward net time ="<<std::chrono::duration_cast<std::chrono::nanoseconds>(t4-t3).count()<<std::endl;
    return result;
}

std::vector<double> get_directed_NN_COST(SharedEnvironment* env, std::vector<Int4>& flow, std::shared_ptr<Network> network_ptr, int all_flow, 
    s_node* curr, std::vector<std::pair<bool, std::vector<double>>>& nn_cached_value
){
    std::vector<double> result(4, 0.0);
    for (int d=0; d<4; ++d){
        std::vector<double> flow_input;
        std::vector<double> map_input;
        std::vector<int> ids;

        get_directed_NN_input(env, flow, all_flow, curr->id, flow_input, map_input, ids, d);
        result[d] = network_ptr->forward(flow_input, map_input)[0];
    }
    return result;
}

#ifdef FOCAL_SEARCH
void update_focal(pqueue_min_f& open, pqueue_min_jam& focal, int& f_min, int& f_bound){
    if(open.empty())
        return;
    if (focal.empty()){
        f_min = open.top()->get_f();
        f_bound = f_min * FOCAL_SEARCH;
    }

    while(!open.empty() && open.top()->get_f() <= f_bound){
        focal.push(open.top());
        open.pop();
    }
    return;
}
#endif
s_node singleShortestPath(SharedEnvironment* env, std::vector<Int4>& flow,
    HeuristicTable& ht,std::vector<int>& traffic, Traj& traj,
    MemoryPool& mem, int start, int goal)
{
    traj.clear();
    traj.push_back(start);
    int neighbors[4];

    while(traj.back() != goal){
        int curr = traj.back();
        getNeighborLocs(env,neighbors,curr);
        int min_h = INT_MAX;
        int next = -1;
        for(int i=0;i<4;i++){
            if(neighbors[i] == -1)
                continue;
            assert(!ht.empty());
            int h = get_heuristic(ht,env, traffic, flow, neighbors[i]);
            if(h < min_h){
                min_h = h;
                next = neighbors[i];
            }
        }
        traj.push_back(next);
    }
    return s_node(goal,traj.size(),0,0,traj.size());
}


s_node aStarOF(SharedEnvironment* env, std::vector<Int4>& flow, int flow_sum, 
    HeuristicTable& ht,std::vector<int>& traffic, Traj& traj,
    MemoryPool& mem, int start, int goal, 
    std::shared_ptr<Network> network_ptr, 
    std::vector<std::pair<bool, std::vector<double>>>& nn_cached_value)
{
    int all_flow = get_all_flow_for_nn(flow_sum);
    // std::cout << "all_flow =" <<all_flow<<std::endl;
    
    // TimePoint start_time = std::chrono::steady_clock::now();
    mem.reset();
    // t+=std::chrono::steady_clock::now()-start_time;
    // cout<<"reset time:"<<t.count()<<endl;


    int expanded=0;
    int generated=0;
    int h;

    //  s_node* root = mem.generate_node(start,0,manhattanDistance(start,goal,env),0,0);
    if(ht.empty())
        h = manhattanDistance(start,goal,env);
    else
        h = get_heuristic(ht,env, traffic, flow, start);
    

    
    s_node* root = mem.generate_node(start,0, h,0,0,0, 0.0);

    if (start == goal){
        traj.clear();
        traj.push_back(start);
        return *root;
    }

    #ifdef FOCAL_SEARCH
    int f_min = h;
    int f_bound = f_min * FOCAL_SEARCH;
    pqueue_min_f open;
    pqueue_min_jam focal;
    re_f ref;
    re_jam rej;

    #else
    pqueue_min_of open;
    re_of re;
    #endif

    open.push(root);



    

    int diff, d, op_flow, total_cross, vertex_flow, depth,p_diff;
    double cost, all_vertex_flow, nn_cost;
    int next_d1, next_d2, next_d1_loc, next_d2_loc;
    int temp_op, temp_vertex;
    double temp_nn_cost;
    double tie_breaker;

    s_node* goal_node = nullptr;
    int neighbors[4];
    int next_neighbors[4];



#ifdef FOCAL_SEARCH
    while (open.size() + focal.size()  > 0){
        update_focal(open,focal,f_min,f_bound);
        s_node* curr = focal.pop();

#else
    while (open.size() > 0){

        s_node* curr = open.pop();

#endif
        // std::cout <<"expanding id ="<< curr->id <<std::endl;
        curr->close();

        if (curr->id == goal){
            goal_node = curr;
            break;
        }
        expanded++;


        // std::cout<<curr->id<<":"<< curr->get_f()<<","<< curr->get_h() <<","<< curr->get_op_flow()<<"," << curr->get_all_vertex_flow()<<"," << std::endl;
        getNeighborLocs(env,neighbors,curr->id);

        std::vector<double> neighbors_nn_cost(4, 0.0);

        // compute weights by network
        if (OBJECTIVE==OBJ::NN){
            if (network_config.use_rotate_input){ 
                // use same weights for all neighbors. 
                // i.e. for all next in Neighbors(curr), w(curr, next) is the same
                neighbors_nn_cost = get_directed_NN_COST(env, flow, network_ptr, all_flow, curr, nn_cached_value);
            } else { 
                // default
                neighbors_nn_cost = get_curr_NN_COST(env, flow, network_ptr, all_flow, curr, nn_cached_value);
            }
        }
        for (int i=0; i<4; i++){
            int next = neighbors[i];
            if (next == -1){
                continue;
            }

            cost = curr->g+1;
            tie_breaker = curr->tie_breaker;


            if (traffic[next] != -1 ){
                int candidates[4] = { next + 1,next + env->cols, next - 1, next - env->cols};
                if (curr->id  == candidates[traffic[next]])
					continue;            
            }

            assert(next >= 0 && next < env->map.size());
            depth = curr->depth + 1;

            //moving direction
            //flow
            op_flow = curr->op_flow; //op_flow is contra flow
            all_vertex_flow = curr->all_vertex_flow;
            nn_cost = curr->nn_cost;

            if(ht.empty())
                h = manhattanDistance(next,goal,env);
            else
                h = get_heuristic(ht,env, traffic, flow, next);
            // std::cout << "id ="<< next << ", h value =" << h <<std::endl;

            diff = next - curr->id;
            d = get_d(diff,env);


            temp_op = ( (flow[curr->id].d[d]+1) * flow[next].d[(d+2)%4]);///( ( (flow[curr->id].d[d]+1) + flow[next].d[(d+2)%4]));
            // std::cout <<"e_cost:" <<flow[curr->id].d[d]+1 << "x"<<flow[next].d[(d+2)%4]<<std::endl;

            //all vertex flow
            //the sum of all out going edge flow is the same as the total number of vertex visiting.
            temp_vertex = 1;
            // std::cout <<"v_cost:";
            for (int j=0; j<4; j++){
                temp_vertex += flow[next].d[j];  
                // std::cout <<flow[next].d[j];              
            }
            // std::cout << std::endl;

            if (OBJECTIVE == OBJ::O_VC){
                op_flow += temp_op;
            }

            // temp_nn_cost = get_NN_COST(env, flow, network_ptr, all_flow, next, curr);
            temp_nn_cost = neighbors_nn_cost[d];
            // std::cout << "d="<<d<<", v_cost ="<<temp_vertex<<", e_cost="<<temp_op<<std::endl;
            
            if (OBJECTIVE == OBJ::NN){
                nn_cost += temp_nn_cost;
            }

#ifdef FOCAL_SEARCH
            if (OBJECTIVE == OBJ::O_VC || OBJECTIVE == OBJ::VC ){
                all_vertex_flow+= (temp_vertex-1) /2;
            }

            if (OBJECTIVE == OBJ::SUM_OVC){
                all_vertex_flow  += (temp_vertex-1) /2 + temp_op;
            }
            if (OBJECTIVE == OBJ::NN) {
                all_vertex_flow += temp_nn_cost;
            }
            if (OBJECTIVE == OBJ::OFFLINE){
                w_id = curr->id * 4 + d;
                all_vertex_flow += env->map_weights[w_id];
            }
#else
            if (OBJECTIVE == OBJ::O_VC || OBJECTIVE == OBJ::VC){
                cost+= (temp_vertex-1) /2;
            }

            if (OBJECTIVE == OBJ::SUM_OVC){
                cost = cost + (temp_vertex-1) /2 + temp_op;
            }

            if (OBJECTIVE == OBJ::NN) {
                cost += temp_nn_cost;
                // std::cout << "nn cost: "<< temp_nn_cost <<std::endl;
                // std::cout << "sum-ovc: "<< (temp_vertex-1) /2 + temp_op <<std::endl;
                // std::cout << "========== [id ="<<next<<"]cost = "<< cost <<"============"<< std::endl;
            }
            if (OBJECTIVE == OBJ::OFFLINE){
                int w_id = curr->id * 4 + d;
                cost += env->map_weights[w_id];
                // std::cout << env->map_weights[w_id] <<std::endl;
            }
#endif

            if(OBJECTIVE == OBJ::SUI_TG){
                tie_breaker = (0.5 * (double)temp_vertex/(double)env->num_of_agents + 0.5 * (double)flow[next].d[(d+2)%4]/(double)env->num_of_agents);
            }

            if(OBJECTIVE == OBJ::SUI_TC){
                tie_breaker += (0.5 * (double)temp_vertex/(double)env->num_of_agents + 0.5 * (double)flow[next].d[(d+2)%4]/(double)env->num_of_agents)/(double)env->max_h;
            }

            p_diff = 0;
            if (curr->parent != nullptr){
                p_diff = curr->id - curr->parent->id;
            }


            s_node temp_node(next,cost,h,op_flow, depth);
            temp_node.tie_breaker = tie_breaker;
            temp_node.set_all_flow(op_flow,  all_vertex_flow);
            temp_node.set_nn_cost(nn_cost);

            if (!mem.has_node(next)){
                s_node* next_node = mem.generate_node(next,cost,h,op_flow, depth,all_vertex_flow, temp_nn_cost);
                next_node->parent = curr;
                next_node->tie_breaker = tie_breaker;
#ifdef FOCAL_SEARCH
                if (next_node->get_f() <= f_bound)
                    focal.push(next_node);
                else
                    open.push(next_node);

#else
                open.push(next_node);
#endif
                generated++;
            }
            else{ 
                s_node* existing = mem.get_node(next);

                if (!existing->is_closed()){

#ifdef FOCAL_SEARCH
                    if (existing->get_f() <= f_bound){
                        //in focal list
                        if (rej(temp_node,*existing)){
                            existing->g = cost;
                            existing->parent = curr;
                            existing->depth = depth;
                            existing->tie_breaker = tie_breaker;
                            existing->set_all_flow(op_flow,  all_vertex_flow);
                            existing->set_nn_cost(nn_cost);
                            focal.decrease_key(existing);
                        }
                    }
                    else{
                        //not in focal list
                        if (ref(temp_node,*existing)){
                            existing->g = cost;
                            existing->parent = curr;
                            existing->depth = depth;
                            existing->tie_breaker = tie_breaker;
                            existing->set_all_flow(op_flow,  all_vertex_flow);
                            existing->set_nn_cost(nn_cost);
                            open.decrease_key(existing);
                            
                        }
                    }

#else
                    if (re(temp_node,*existing)){
                        existing->g = cost;
                        existing->parent = curr;
                        existing->depth = depth;
                        existing->tie_breaker = tie_breaker;
                        existing->set_all_flow(op_flow,  all_vertex_flow);
                        existing->set_nn_cost(nn_cost);
                        open.decrease_key(existing);
                    }
#endif
                }
                else{
                    // closed , check if re expansion needed
#ifdef FOCAL_SEARCH
                    // if (ref(temp_node,*existing)){
                    //     existing->g = cost;
                    //     existing->parent = curr;
                    //     existing->depth = depth;
                    //     existing->set_all_flow(op_flow,  all_vertex_flow);
                    //     if (existing->get_f() <= f_bound){
                    //         focal.push(existing);
                    //     }
                    //     else{
                    //         open.push(existing);
                    //     }
                    // }
#else

                if (re(temp_node,*existing)){ 
                    std::cout << "error in aStarOF: re-expansion" << std::endl;
                    assert(false);
                    exit(1);
                }
#endif

                } 
            }
        }
            

          
    }

    // std::cout << "expanded: " << expanded << std::endl;
    // std::cout << "generated: " << generated << std::endl;
    if (goal_node == nullptr){
        std::cout << "error in aStarOF: no path found "<< start<<","<<goal << std::endl;
        assert(false);
        exit(1);
    }

    traj.resize(goal_node->depth);
    s_node* curr = goal_node;
    for (int i=goal_node->depth-1; i>=0; i--){
        traj[i] = curr->id;
        curr = curr->parent;
    }
    // std::cout<< goal_node->get_f()<<","<< goal_node->get_h() <<","<< goal_node->get_op_flow()<<"," << goal_node->get_all_vertex_flow()<<"," << std::endl;
    // std::cout<< root->get_f()<<","<< root->get_h()<<","<< root->get_op_flow()<<"," << root->get_all_vertex_flow()<<"," << std::endl;
    // exit(1);
    return *goal_node;
}
}

