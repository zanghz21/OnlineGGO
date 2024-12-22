#include "SingleAgentSolver.h"

double SingleAgentSolver::compute_h_value(
    const BasicGraph &G, int curr, int goal_id,
    const vector<tuple<int, int, int>> &goal_location) const
{
    // std::cout << "begin"<<std::endl;
    int goal_loc = std::get<0>(goal_location[goal_id]);
    if (G.heuristics.find(goal_loc) == G.heuristics.end()){
        std::cout<< "error goal_loc ="<<goal_loc<<", not find in G.h"<<std::endl;
        exit(1);
    }
    if (G.heuristics.at(goal_loc).size() <= curr){
        std::cout << "error curr loc ="<<curr<<", larger than "<<G.heuristics.at(goal_loc).size()<<std::endl;
        exit(1);
    }
    double h = G.heuristics.at(goal_loc)[curr];
    goal_id++;
    while (goal_id < (int)goal_location.size())
    {
        if (G.heuristics.find(std::get<0>(goal_location[goal_id])) == G.heuristics.end()){
            std::cout<< "error goal_loc ="<<goal_loc<<", not find in G.h"<<std::endl;
            exit(1);
        }
        if (G.heuristics.at(std::get<0>(goal_location[goal_id])).size() <= std::get<0>(goal_location[goal_id - 1])){
            std::cout << "error curr loc ="<<curr<<", larger than "<<G.heuristics.at(goal_loc).size()<<std::endl;
            exit(1);
        }
        h += G.heuristics.at(
                std::get<0>(goal_location[goal_id])
            )[std::get<0>(goal_location[goal_id - 1])];
        goal_id++;
    }
    // std::cout << "end, h = "<<h<<std::endl;
    return h;
}
