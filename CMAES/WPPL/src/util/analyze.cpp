#include "common.h"
#include "nlohmann/json.hpp"
#include "Grid.h"

int get_orient_idx(string o) {
    if (o=="E") return 0;
    if (o=="S") return 1;
    if (o=="W") return 2;
    if (o=="N") return 3;
    std::cerr<<"unknown orientation: "<<o<<std::endl;
    exit(-1);
}

double get_sum(std::vector<double> v)
{
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum;
}

std::tuple<double, double> get_mean_std(std::vector<double> v)
{
    if (v.size() == 0)
        return std::make_tuple(0, 0);
    double sum = get_sum(v);
    double mean = sum / v.size();

    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(),
                    [mean](double x)
                    { return x - mean; });

    double sq_sum = std::inner_product(diff.begin(), diff.end(),
                                        diff.begin(), 0.0);
    double sigma = sqrt(sq_sum / v.size());
    return std::make_tuple(mean, sigma);
}


int get_Manhattan_distance(int loc1, int loc2, int cols) {
    return abs(loc1 / cols - loc2 / cols) + abs(loc1 % cols - loc2 % cols);
}

tuple<vector<double>, double, double> edge_pair_usage_mean_std(Grid & grid, vector<vector<double>> &edge_usage)
{
    // For each pair of valid edges (i, j) and (j, i), calculate the absolute
    // of their edge usage difference, and calculate mean and std.
    vector<double> edge_pair_usage;
    int valid_edge_id = 0;
    int n_vertices = grid.rows * grid.cols;
    for (int i = 0; i < n_vertices; i++)
	{
        // Start from i+1 to ignore wait actions
        for (int j = i + 1; j < n_vertices; j++)
        {
            if (grid.map[i] ==0 && grid.map[j] == 0 &&
                get_Manhattan_distance(i, j, grid.cols) <= 1)
            {
                edge_pair_usage.push_back(abs(edge_usage[i][j] - edge_usage[j][i]));
                valid_edge_id += 1;
            }
        }
	}
    // cout << "Number of valid edge pairs: " << edge_pair_usage.size() << endl;
    // cout << "End of valid edge pair counter: " << valid_edge_id << endl;
    double mean, std;
    std::tie(mean, std) = get_mean_std(edge_pair_usage);
    return make_tuple(edge_pair_usage, mean, std);
}


#ifndef NO_ROT
void move(int & x,int & y,int & o, char action) {
    if (action=='F') {
        if (o==0) {
            x+=1;
        } else if (o==1) {
            y+=1;
        } else if (o==2) {
            x-=1;
        } else if (o==3) {
            y-=1;
        } else {
            std::cerr<<"unknown orientaiton: "<<o<<std::endl;
            exit(-1);
        }
    } else if (action=='R') {
        o=(o+1)%4;
    } else if (action=='C') {
        o=(o+3)%4;
    } else if (action=='W') {
        // do nothing
    } else {
        std::cerr<<"unknown action: "<<action<<std::endl;
        exit(-1);
    }

}

nlohmann::json analyze_result_json(const nlohmann::json & result, Grid & grid) {

    int h=grid.rows;
    int w=grid.cols;

    // objective: throughput
    double throughput=result["numTaskFinished"].get<double>();
    auto actions_str = result["actualPaths"][0].get<std::string>();
    int T=actions_str.size()/2+1;
    // std::cout<<"T: "<<T<<std::endl;

    double avg_throughput=throughput/T;
    // std::cout<<"total throughput: "<<throughput<<" avg throughput:"<<avg_throughput<<std::endl;

    // statistics:
    // 1. vertex usage
    // 2. edge usage
    // 3. 
    int map_size=h*w;
    std::vector<double> vertex_usage(map_size,0);
    std::vector<std::vector<double> > edge_usage(map_size, std::vector<double>(map_size,0));

    // NOTE: this format is different from what we used in c++ code: right, up, left, down
    std::vector<double> edge_usage_matrix(map_size*4,0);
    std::vector<double> vertex_wait_matrix(map_size,0);

    int team_size=result["teamSize"].get<int>();
    // std::cout<<"team size: "<<team_size<<std::endl;

    for (int aid=0;aid<team_size;++aid) {
        auto actions_str = result["actualPaths"][aid].get<std::string>();
        // std::cerr<<"agent "<<aid<<" actions: "<<actions_str<<std::endl;

        auto start_y = result["start"][aid][0].get<int>();
        auto start_x = result["start"][aid][1].get<int>();
        auto start_orient = get_orient_idx(result["start"][aid][2].get<std::string>());
        // std::cerr<<"agent "<<aid<<" start: "<<start_y<<","<<start_x<<","<<start_orient<<std::endl;

        auto pos=start_y*w+start_x;
        // update vertex usage
        vertex_usage[pos]+=1;

        // simulate
        auto prev_x=start_x;
        auto prev_y=start_y;
        auto prev_orient=start_orient;
        for (int i=0;i<actions_str.size();i=i+2) {
            // std::cerr<<"iter "<<i<<std::endl;
            char action=actions_str[i];
            auto curr_x=prev_x;
            auto curr_y=prev_y;
            auto curr_orient=prev_orient;
            move(curr_x,curr_y,curr_orient,action);

            if (curr_x<0 || curr_x>=w || curr_y<0 || curr_y>=h) {
                std::cerr<<"agent "<<aid<<" out of bound"<<std::endl;
                exit(-1);
            }

            auto prev_pos=prev_y*w+prev_x;
            auto curr_pos=curr_y*w+curr_x;

            if (action=='F') {
                if (prev_orient==0) { // right
                    edge_usage_matrix[prev_pos*4+0]+=1;
                } else if (prev_orient==1) { // down
                    edge_usage_matrix[prev_pos*4+3]+=1;
                } else if (prev_orient==2) { // left
                    edge_usage_matrix[prev_pos*4+2]+=1;
                } else if (prev_orient==3) { // up
                    edge_usage_matrix[prev_pos*4+1]+=1;
                } else {
                    std::cerr<<"unknown orientation: "<<prev_orient<<std::endl;
                    exit(-1);
                }
            }
            else if (action=='W') {
                vertex_wait_matrix[prev_pos]+=1;
            } 

            // update vertex usage
            vertex_usage[curr_pos]+=1;
            // update edge usage
            edge_usage[prev_pos][curr_pos]+=1;

            prev_x=curr_x;
            prev_y=curr_y;
            prev_orient=curr_orient;
        }
    }

    double edge_pair_usage_mean, edge_pair_usage_std;
    vector<double> edge_pair_usage;
    std::tie(edge_pair_usage, edge_pair_usage_mean, edge_pair_usage_std) = edge_pair_usage_mean_std(grid, edge_usage);

    
    nlohmann::json exec_moves =nlohmann::json::array();
    for(int aid=0; aid<team_size; ++aid){
        std::string exec_m;
        int prev_h=-1, prev_w=-1;
        for (auto loc: result["execFuture"][aid]){
            int h = loc[0].get<int>();
            int w = loc[1].get<int>();
            // std::cout << h<<", "<<w<<std::endl;
            if(prev_h==-1){
                prev_h = h;
                prev_w = w;
                continue;
            }
            
            if((h==prev_h)&&(w==prev_w)){
                exec_m += "W";
            } else if((h==prev_h)&&(w==prev_w+1)){
                exec_m += "R";
            } else if ((h==prev_h)&&(w==prev_w-1)){
                exec_m += "L";
            } else if ((h==prev_h+1)&&(w==prev_w)){
                exec_m += "D";
            } else if ((h==prev_h-1)&&(w==prev_w)){
                exec_m += "U";
            } else {
                std::cout << prev_h <<", "<<prev_w<<", "<<h<<","<<w<<std::endl;
                std::cout << "must have bug in exec_move" <<std::endl;
                exit(-1);
            }
            prev_h = h;
            prev_w = w;
        }
        exec_moves.push_back(exec_m);
    }

    nlohmann::json plan_moves =nlohmann::json::array();
    for(int aid=0; aid<team_size; ++aid){
        int prev_h=-1, prev_w=-1;
        std::string plan_m;
        for (auto loc: result["planFuture"][aid]){
            int h = loc[0].get<int>();
            int w = loc[1].get<int>();
            if(prev_h==-1){
                prev_h = h;
                prev_w = w;
                continue;
            }
            
            if((h==prev_h)&&(w==prev_w)){
                plan_m += "W";
            } else if((h==prev_h)&&(w==prev_w+1)){
                plan_m += "R";
            } else if ((h==prev_h)&&(w==prev_w-1)){
                plan_m += "L";
            } else if ((h==prev_h+1)&&(w==prev_w)){
                plan_m += "D";
            } else if ((h==prev_h-1)&&(w==prev_w)){
                plan_m += "U";
            } else {
                std::cout << prev_h <<", "<<prev_w<<", "<<h<<","<<w<<std::endl;
                std::cout << "must have bug in plan_move" <<std::endl;
                exit(-1);
            }
            prev_h = h;
            prev_w = w;
        }
        plan_moves.push_back(plan_m);
    }

    nlohmann::json analysis;
    analysis = {
        {"num_task_finished", throughput}, 
        {"throughput", avg_throughput},
        {"tile_usage", vertex_usage},
        {"edge_pair_usage", edge_pair_usage},
        {"edge_pair_usage_mean", edge_pair_usage_mean},
        {"edge_pair_usage_std", edge_pair_usage_std},
        {"edge_usage_matrix", edge_usage_matrix},
        {"vertex_wait_matrix", vertex_wait_matrix}, 
        {"final_pos", result["final_pos"]}, 
        {"final_tasks", result["final_tasks"]}, 
        {"actual_paths", result["actualPaths"]}, 
        {"starts", result["start"]}, 
        {"exec_future", result["execFuture"]}, 
        {"plan_future", result["planFuture"]}, 
        {"exec_move", exec_moves}, 
        {"plan_move", plan_moves}
    };
    return analysis;
}

nlohmann::json analyze_curr_result_json(const nlohmann::json & result, Grid & grid) {

    // objective: throughput
    double throughput=result["numTaskFinishedSoFar"].get<double>();
    auto actions_str = result["actualPaths"][0].get<std::string>();
    int T=actions_str.size()/2+1;

    double avg_throughput=throughput/T;

    int team_size=result["teamSize"].get<int>();

    std::cout << 1<<std::endl;
    nlohmann::json exec_moves =nlohmann::json::array();
    for(int aid=0; aid<team_size; ++aid){
        std::string exec_m;
        int prev_h=-1, prev_w=-1;
        for (auto loc: result["execFuture"][aid]){
            int h = loc[0].get<int>();
            int w = loc[1].get<int>();
            // std::cout << h<<", "<<w<<std::endl;
            if(prev_h==-1){
                prev_h = h;
                prev_w = w;
                continue;
            }
            
            if((h==prev_h)&&(w==prev_w)){
                exec_m += "W";
            } else if((h==prev_h)&&(w==prev_w+1)){
                exec_m += "R";
            } else if ((h==prev_h)&&(w==prev_w-1)){
                exec_m += "L";
            } else if ((h==prev_h+1)&&(w==prev_w)){
                exec_m += "D";
            } else if ((h==prev_h-1)&&(w==prev_w)){
                exec_m += "U";
            } else {
                std::cout << prev_h <<", "<<prev_w<<", "<<h<<","<<w<<std::endl;
                std::cout << "must have bug in exec_move" <<std::endl;
                exit(-1);
            }
            prev_h = h;
            prev_w = w;
        }
        exec_moves.push_back(exec_m);
    }
std::cout << 2<<std::endl;
    nlohmann::json plan_moves =nlohmann::json::array();
    for(int aid=0; aid<team_size; ++aid){
        int prev_h=-1, prev_w=-1;
        std::string plan_m;
        for (auto loc: result["planFuture"][aid]){
            int h = loc[0].get<int>();
            int w = loc[1].get<int>();
            if(prev_h==-1){
                prev_h = h;
                prev_w = w;
                continue;
            }
            
            if((h==prev_h)&&(w==prev_w)){
                plan_m += "W";
            } else if((h==prev_h)&&(w==prev_w+1)){
                plan_m += "R";
            } else if ((h==prev_h)&&(w==prev_w-1)){
                plan_m += "L";
            } else if ((h==prev_h+1)&&(w==prev_w)){
                plan_m += "D";
            } else if ((h==prev_h-1)&&(w==prev_w)){
                plan_m += "U";
            } else {
                std::cout << prev_h <<", "<<prev_w<<", "<<h<<","<<w<<std::endl;
                std::cout << "must have bug in plan_move" <<std::endl;
                exit(-1);
            }
            prev_h = h;
            prev_w = w;
        }
        plan_moves.push_back(plan_m);
    }

    nlohmann::json analysis;
    analysis = {
        {"num_task_finished", throughput}, 
        {"throughput", avg_throughput},
        {"final_pos", result["final_pos"]}, 
        {"final_tasks", result["final_tasks"]}, 
        {"actual_paths", result["actualPaths"]}, 
        {"starts", result["start"]}, 
        {"exec_future", result["execFuture"]}, 
        {"plan_future", result["planFuture"]}, 
        {"exec_move", exec_moves}, 
        {"plan_move", plan_moves}
    };
    return analysis;
}


#else 

void move(int & x,int & y, char action) {
    if (action=='R') {
        x+=1;
    } else if (action=='D') {
        y+=1;
    } else if (action=='L') {
        x-=1;
    } else if (action=='U') {
        y-=1;
    } else if (action=='W') {
        // do nothing
    } else {
        std::cerr<<"unknown action: "<<action<<std::endl;
        exit(-1);
    }

}

nlohmann::json analyze_curr_result_json(const nlohmann::json & result, Grid & grid) {

    int H=grid.rows;
    int W=grid.cols;

    // objective: throughput
    double throughput=result["numTaskFinishedSoFar"].get<double>();
    auto actions_str = result["actualPaths"][0].get<std::string>();
    int T=actions_str.size()/2+1;
    // std::cout<<"T: "<<T<<std::endl;

    double avg_throughput=throughput/T;
    // std::cout<<"total throughput: "<<throughput<<" avg throughput:"<<avg_throughput<<std::endl;

    int team_size=result["teamSize"].get<int>();
    // std::cout<<"team size: "<<team_size<<std::endl;
    nlohmann::json past_paths =nlohmann::json::array();
    for (int aid=0;aid<team_size;++aid) {
        nlohmann::json agent_past_path = nlohmann::json::array();

        auto actions_str = result["actualPaths"][aid].get<std::string>();
        // std::cerr<<"agent "<<aid<<" actions: "<<actions_str<<std::endl;

        auto start_y = result["start"][aid][0].get<int>();
        auto start_x = result["start"][aid][1].get<int>();
        // std::cerr<<"agent "<<aid<<" start: "<<start_y<<","<<start_x<<std::endl;

        auto pos=start_y*W+start_x;

        // simulate
        auto prev_x=start_x;
        auto prev_y=start_y;
        for (int i=0;i<actions_str.size();i=i+2) {
            // std::cerr<<"iter "<<i<<std::endl;
            agent_past_path.push_back(nlohmann::json::array({prev_y, prev_x}));
            char action=actions_str[i];
            auto curr_x=prev_x;
            auto curr_y=prev_y;
            move(curr_x,curr_y,action);

            if (curr_x<0 || curr_x>=W || curr_y<0 || curr_y>=H) {
                std::cerr<<"agent "<<aid<<" out of bound"<<std::endl;
                exit(-1);
            }
            prev_x=curr_x;
            prev_y=curr_y;
        }
        agent_past_path.push_back(nlohmann::json::array({prev_y, prev_x}));
        past_paths.push_back(agent_past_path);
    }
    nlohmann::json exec_moves =nlohmann::json::array();
    for(int aid=0; aid<team_size; ++aid){
        std::string exec_m;
        int prev_h=-1, prev_w=-1;
        for (auto loc: result["execFuture"][aid]){
            int h = loc[0].get<int>();
            int w = loc[1].get<int>();
            // std::cout << h<<", "<<w<<std::endl;
            if(prev_h==-1){
                prev_h = h;
                prev_w = w;
                continue;
            }
            
            if((h==prev_h)&&(w==prev_w)){
                exec_m += "W";
            } else if((h==prev_h)&&(w==prev_w+1)){
                exec_m += "R";
            } else if ((h==prev_h)&&(w==prev_w-1)){
                exec_m += "L";
            } else if ((h==prev_h+1)&&(w==prev_w)){
                exec_m += "D";
            } else if ((h==prev_h-1)&&(w==prev_w)){
                exec_m += "U";
            } else {
                std::cout << prev_h <<", "<<prev_w<<", "<<h<<","<<w<<std::endl;
                std::cout << "must have bug in exec_move" <<std::endl;
                exit(-1);
            }
            prev_h = h;
            prev_w = w;
        }
        exec_moves.push_back(exec_m);
    }
    
    nlohmann::json plan_moves = nlohmann::json::array();
    for(int aid=0; aid<team_size; ++aid){
        int prev_h=-1, prev_w=-1;
        std::string plan_m;
        for (auto loc: result["planFuture"][aid]){
            int h = loc[0].get<int>();
            int w = loc[1].get<int>();
            if(prev_h==-1){
                prev_h = h;
                prev_w = w;
                continue;
            }
            
            if((h==prev_h)&&(w==prev_w)){
                plan_m += "W";
            } else if((h==prev_h)&&(w==prev_w+1)){
                plan_m += "R";
            } else if ((h==prev_h)&&(w==prev_w-1)){
                plan_m += "L";
            } else if ((h==prev_h+1)&&(w==prev_w)){
                plan_m += "D";
            } else if ((h==prev_h-1)&&(w==prev_w)){
                plan_m += "U";
            } else {
                std::cout << prev_h <<", "<<prev_w<<", "<<h<<","<<w<<std::endl;
                std::cout << "must have bug in plan_move" <<std::endl;
                exit(-1);
            }
            prev_h = h;
            prev_w = w;
        }
        plan_moves.push_back(plan_m);
    }

    nlohmann::json analysis;
    analysis = {
        {"num_task_finished", throughput}, 
        {"throughput", avg_throughput},
        {"final_pos", result["final_pos"]}, 
        {"final_tasks", result["final_tasks"]}, 
        {"actual_paths", result["actualPaths"]}, 
        {"starts", result["start"]}, 
        {"exec_future", result["execFuture"]}, 
        {"plan_future", result["planFuture"]}, 
        {"exec_move", exec_moves}, 
        {"plan_move", plan_moves}, 
        {"past_paths", past_paths}, 
        {"done", result["done"]}, 
        {"agents_finish_task", result["agents_finish_task"]}
    };
    return analysis;
}

nlohmann::json analyze_result_json(const nlohmann::json & result, Grid & grid) {

    int h=grid.rows;
    int w=grid.cols;

    // objective: throughput
    double throughput=result["numTaskFinished"].get<double>();
    auto actions_str = result["actualPaths"][0].get<std::string>();
    int T=actions_str.size()/2+1;
    // std::cout<<"T: "<<T<<std::endl;

    double avg_throughput=throughput/T;
    // std::cout<<"total throughput: "<<throughput<<" avg throughput:"<<avg_throughput<<std::endl;

    // statistics:
    // 1. vertex usage
    // 2. edge usage
    // 3. 
    int map_size=h*w;
    std::vector<double> vertex_usage(map_size,0);
    std::vector<std::vector<double> > edge_usage(map_size, std::vector<double>(map_size,0));

    // NOTE: this format is different from what we used in c++ code: right, up, left, down
    std::vector<double> edge_usage_matrix(map_size*4,0);
    std::vector<double> vertex_wait_matrix(map_size,0);

    int team_size=result["teamSize"].get<int>();
    // std::cout<<"team size: "<<team_size<<std::endl;

    nlohmann::json past_paths =nlohmann::json::array();
    for (int aid=0;aid<team_size;++aid) {
        nlohmann::json agent_past_path = nlohmann::json::array();

        auto actions_str = result["actualPaths"][aid].get<std::string>();
        // std::cerr<<"agent "<<aid<<" actions: "<<actions_str<<std::endl;

        auto start_y = result["start"][aid][0].get<int>();
        auto start_x = result["start"][aid][1].get<int>();
        // std::cerr<<"agent "<<aid<<" start: "<<start_y<<","<<start_x<<","<<start_orient<<std::endl;

        auto pos=start_y*w+start_x;
        // update vertex usage
        vertex_usage[pos]+=1;

        // simulate
        auto prev_x=start_x;
        auto prev_y=start_y;
        for (int i=0;i<actions_str.size();i=i+2) {
            // std::cerr<<"iter "<<i<<std::endl;
            agent_past_path.push_back(nlohmann::json::array({prev_y, prev_x}));
            char action=actions_str[i];
            auto curr_x=prev_x;
            auto curr_y=prev_y;
            move(curr_x,curr_y,action);

            if (curr_x<0 || curr_x>=w || curr_y<0 || curr_y>=h) {
                std::cerr<<"agent "<<aid<<" out of bound"<<std::endl;
                exit(-1);
            }

            auto prev_pos=prev_y*w+prev_x;
            auto curr_pos=curr_y*w+curr_x;

            if (action=='R') {
                edge_usage_matrix[prev_pos*4+0]+=1;
            } else if (action=='D') {
                edge_usage_matrix[prev_pos*4+3]+=1;                
            } else if (action=='L') {
                edge_usage_matrix[prev_pos*4+2]+=1;
            } else if (action=='U') {
                edge_usage_matrix[prev_pos*4+1]+=1;
            } else if (action=='W') {
                vertex_wait_matrix[prev_pos]+=1;
            } 

            // update vertex usage
            vertex_usage[curr_pos]+=1;
            // update edge usage
            edge_usage[prev_pos][curr_pos]+=1;

            prev_x=curr_x;
            prev_y=curr_y;
        }
        agent_past_path.push_back(nlohmann::json::array({prev_y, prev_x}));
        past_paths.push_back(agent_past_path);
    }

    double edge_pair_usage_mean, edge_pair_usage_std;
    vector<double> edge_pair_usage;
    std::tie(edge_pair_usage, edge_pair_usage_mean, edge_pair_usage_std) = edge_pair_usage_mean_std(grid, edge_usage);

    
    nlohmann::json exec_moves =nlohmann::json::array();
    for(int aid=0; aid<team_size; ++aid){
        std::string exec_m;
        int prev_h=-1, prev_w=-1;
        for (auto loc: result["execFuture"][aid]){
            int h = loc[0].get<int>();
            int w = loc[1].get<int>();
            // std::cout << h<<", "<<w<<std::endl;
            if(prev_h==-1){
                prev_h = h;
                prev_w = w;
                continue;
            }
            
            if((h==prev_h)&&(w==prev_w)){
                exec_m += "W";
            } else if((h==prev_h)&&(w==prev_w+1)){
                exec_m += "R";
            } else if ((h==prev_h)&&(w==prev_w-1)){
                exec_m += "L";
            } else if ((h==prev_h+1)&&(w==prev_w)){
                exec_m += "D";
            } else if ((h==prev_h-1)&&(w==prev_w)){
                exec_m += "U";
            } else {
                std::cout << prev_h <<", "<<prev_w<<", "<<h<<","<<w<<std::endl;
                std::cout << "must have bug in exec_move" <<std::endl;
                exit(-1);
            }
            prev_h = h;
            prev_w = w;
        }
        exec_moves.push_back(exec_m);
    }
    
    nlohmann::json plan_moves = nlohmann::json::array();
    for(int aid=0; aid<team_size; ++aid){
        int prev_h=-1, prev_w=-1;
        std::string plan_m;
        for (auto loc: result["planFuture"][aid]){
            int h = loc[0].get<int>();
            int w = loc[1].get<int>();
            if(prev_h==-1){
                prev_h = h;
                prev_w = w;
                continue;
            }
            
            if((h==prev_h)&&(w==prev_w)){
                plan_m += "W";
            } else if((h==prev_h)&&(w==prev_w+1)){
                plan_m += "R";
            } else if ((h==prev_h)&&(w==prev_w-1)){
                plan_m += "L";
            } else if ((h==prev_h+1)&&(w==prev_w)){
                plan_m += "D";
            } else if ((h==prev_h-1)&&(w==prev_w)){
                plan_m += "U";
            } else {
                std::cout << prev_h <<", "<<prev_w<<", "<<h<<","<<w<<std::endl;
                std::cout << "must have bug in plan_move" <<std::endl;
                exit(-1);
            }
            prev_h = h;
            prev_w = w;
        }
        plan_moves.push_back(plan_m);
    }

    nlohmann::json analysis;
    analysis = {
        {"num_task_finished", throughput}, 
        {"throughput", avg_throughput},
        {"tile_usage", vertex_usage},
        {"edge_pair_usage", edge_pair_usage},
        {"edge_pair_usage_mean", edge_pair_usage_mean},
        {"edge_pair_usage_std", edge_pair_usage_std},
        {"edge_usage_matrix", edge_usage_matrix},
        {"vertex_wait_matrix", vertex_wait_matrix}, 
        {"final_pos", result["final_pos"]}, 
        {"final_tasks", result["final_tasks"]}, 
        {"actual_paths", result["actualPaths"]}, 
        {"starts", result["start"]}, 
        {"exec_future", result["execFuture"]}, 
        {"plan_future", result["planFuture"]}, 
        {"exec_move", exec_moves}, 
        {"plan_move", plan_moves}, 
        {"past_paths", past_paths}, 
        {"done", result["done"]}, 
        {"agents_finish_task", result["agents_finish_task"]}
    };
    return analysis;
}

#endif