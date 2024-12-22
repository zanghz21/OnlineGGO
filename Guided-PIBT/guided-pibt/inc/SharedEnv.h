#pragma once
#include "States.h"
#include "nlohmann/json.hpp"


class SharedEnvironment {
public:
    int num_of_agents;

    int rows;
    int cols;
    std::string map_name;
    std::vector<int> map;
    std::vector<double> map_weights; // [h*w*4], RDLU

    std::vector<std::vector<int>> hist_edge_usage; // [h*w, 5], RDLUW
    int past_traffic_interval=-1; // hist_edge_usage records traffic info in (t - past_traffic_interval, t]

    int max_h=0;

    std::string file_storage_path;

    // goal locations for each agent
    // each task is a pair of <goal_loc, reveal_time>
    vector< vector<pair<int, int> > > goal_locations; // [agents, curr_tasks_seq]
    std::vector<int> goal_loc_arr; // [h*w]
	std::vector<std::vector<int>> neighbors;

    int curr_timestep = 0;
    vector<State> curr_states;

    SharedEnvironment(){}

    void update_past_traffic_interval(int interval){
        this->past_traffic_interval = interval;
    }
    void reset_hist_edge_usage(){
        // std::cout <<"calling reset"<<std::endl;
        // std::cout <<this->hist_edge_usage.size()<<","<<this->cols<<", "<<this->rows<<std::endl;
        this->hist_edge_usage.clear();
        this->hist_edge_usage.resize(this->cols*this->rows, std::vector<int>(5, 0));
        // this->print_hist_edge_usage();
        // std::cout << "end reset, size = "<<this->hist_edge_usage.size()<< std::endl;
    }

    void print_hist_edge_usage(){
        for (int i=0; i<5; ++i){
            std::cout << "i = "<<i<<std::endl;
            for (int r=0; r<this->rows; ++r){
                for (int c=0; c<this->cols; ++c){
                    std::cout << this->hist_edge_usage[r*this->cols+c][i]<<", ";
                }
                std::cout << std::endl;
            }
        }
    }

    void print_goal_arr(){
        for (int r=0; r<this->rows; ++r){
            for (int c=0; c<this->cols; ++c){
                std::cout << this->goal_loc_arr[r*this->cols+c]<<", ";
            }
            std::cout << std::endl;
        }
    }

    void init_neighbor(){
        neighbors.resize(rows * cols);
        for (int row=0; row<rows; row++){
            for (int col=0; col<cols; col++){
                int loc = row*cols+col;
                if (map[loc]==0){
                    if (row>0 && map[loc-cols]==0){
                        neighbors[loc].push_back(loc-cols);
                    }
                    if (row<rows-1 && map[loc+cols]==0){
                        neighbors[loc].push_back(loc+cols);
                    }
                    if (col>0 && map[loc-1]==0){
                        neighbors[loc].push_back(loc-1);
                    }
                    if (col<cols-1 && map[loc+1]==0){
                        neighbors[loc].push_back(loc+1);
                    }
                }
            }
        }
    };

};
