#pragma once
#include <boost/heap/pairing_heap.hpp>
#include <boost/unordered_set.hpp>
#include <vector>
#include <iostream>
#include <cfloat>
#include <memory>

namespace HT_v2 {

constexpr float max_heuristic_val=FLT_MAX/16;

class State {
public:
    int loc;
    float g;
    float h;
    float f;

    State(int _loc, float _g, float _h):
        loc(_loc), g(_g), h(_h), f(_g+_h) {}

    void copy(const State *s) {
        loc = s->loc;
        g = s->g;
        h = s->h;
        f = s->f;
    }

    struct StateHash {
        inline std::size_t operator()(const State *s) const {
            return s->loc;
        }
    };

    struct StateEqualTo {
        inline bool operator()(const State *s1, const State *s2) const {
            return s1->loc==s2->loc;
        }
    };

    struct StateCompare {
        inline bool operator()(const State *s1, const State *s2) const {
            if (s1->f==s2->f) {
                if (s1->h==s2->h) {
                    return s1->h>s2->h;
                }
            }
            return s1->f>s2->f;
        }
    };

    bool closed=false;

    boost::heap::pairing_heap<State *, boost::heap::compare<State::StateCompare> >::handle_type open_handle;
};


class LazyBDHeuristicTable {
private:
    int id;
    std::vector<float> map_weights;
    std::vector<float> newest_map_weights;

    void update_weights_to_newest(){
        // std::cout << "id="<<this->id<<", update to newest, 1st weights =" <<this->newest_map_weights[0]<<", origin 1st ="<<this->map_weights[0]<<std::endl;
        // for (int i=0; i<newest_map_weights.size(); ++i){
        //     this->map_weights[i] = this->newest_map_weights[i];
        // }
        this->map_weights = this->newest_map_weights;
        this->min_map_weight = this->new_min_map_weight;
        this->is_newest_weights = true;
        // std::cout << "update_weights_to_newest end" <<std::endl;
    }
public:

    bool is_newest_weights;
    // should we use map here?
    const std::vector<int> & map;
    int rows;
    int cols;
    
    float min_map_weight;
    float new_min_map_weight;

    std::vector<float> h_vals;
    boost::heap::pairing_heap<State*,boost::heap::compare<State::StateCompare> > open_list;
    boost::unordered_set<State*,State::StateHash,State::StateEqualTo> all_states;

    int start_loc=-1;
    int goal_loc=-1;

    int sy,sx,gy,gx;

    // R,D,L,U
    const int dy[4]={0,1,0,-1};
    const int dx[4]={1,0,-1,0};

    LazyBDHeuristicTable(int _id, const std::vector<int> & _map, int _rows, int _cols, const std::vector<float> _map_weights, float _min_map_weight=-1):
        id(_id), map(_map), rows(_rows), cols(_cols), map_weights(_map_weights), newest_map_weights(_map_weights), 
        h_vals(_rows*_cols, -1), min_map_weight(_min_map_weight), new_min_map_weight(_min_map_weight), is_newest_weights(true) {
        // check map weights' size
        if (map_weights.size()!=_rows*_cols*5) {
            std::cerr<<"map weights' size is not equal to rows*cols*5 (r,d,l,u,wait)"<<std::endl;
            exit(1);
        }

        if (min_map_weight<0) {
            min_map_weight=max_heuristic_val;
            for (size_t i=0;i<map_weights.size();++i) {
                if (map_weights[i]<min_map_weight) {
                    min_map_weight=map_weights[i];
                }
            }
        }

    }

    void print_newest_weights(){
        // for(int i=0; i<this->newest_map_weights.size(); ++i){
        for(int i=0; i<10; ++i){
            float val = this->newest_map_weights[i];
            std::cout << val <<", ";
        }
        std::cout << std::endl;
    }

    void update_newest_weights(std::vector<float> _new_weights, float _new_min_map_weight){
        if (_new_weights.size() != this->newest_map_weights.size()){
            std::cout << "size error!" <<std::endl;
            exit(1);
        }
        
        // for (int i=0; i<_new_weights.size(); ++i){
        //     this->newest_map_weights[i] = _new_weights[i];
        // }
        this->newest_map_weights = _new_weights;
        this->new_min_map_weight = _new_min_map_weight;
    }


    void update_weights_at_once(std::vector<float> _new_weights, float _new_min_map_weight){
        this->update_newest_weights(_new_weights, _new_min_map_weight);
        this->update_weights_to_newest();
        this->reset_ht();
    }

    void reset_ht(){
        // std::cout << "reset ht for id="<<this->id<<", start loc ="<< this->start_loc << ", goal loc ="<<this->goal_loc <<std::endl;
        sy=this->start_loc/cols;
        sx=this->start_loc%cols;
        gy=this->goal_loc/cols;
        gx=this->goal_loc%cols;


        for (size_t i=0;i<h_vals.size();++i) {
            h_vals[i]=-1;
        }

        // clear memory
        open_list.clear();
        for (auto & s: all_states) {
            delete s;
        }
        all_states.clear();

        // push root node
        State *root = new State(goal_loc, 0, get_h(gy,gx));
        root->closed = false;
        root->open_handle=open_list.push(root);
        all_states.insert(root);
    }

    float get_h(int y, int x) {
        return min_map_weight*(std::abs(y-sy)+std::abs(x-sx));
    }

    void update_start_and_goal(int _start_loc, int _goal_loc) {
        // std::cout << "update_start_and_goal" <<std::endl;
        if (goal_loc==_goal_loc) {
            return;
        }

        // std::cout<<"update start and goal to "<<_start_loc<<" "<<_goal_loc<<std::endl;

        if (!this->is_newest_weights){
            this->update_weights_to_newest();
        }

        start_loc=_start_loc;
        goal_loc=_goal_loc;
        this->reset_ht();
    }

    void update_start(int _start_loc){
        // std::cout << "warning: only use when update at once!!" <<std::endl;
        this->start_loc = _start_loc;
    }

    float get(int loc) {
        float h=max_heuristic_val;

        // we always assume all nodes are connected
        if (h_vals[loc]<0) {
            bool found=astar(loc);
            if (!found) {
                int y=loc/cols;
                int x=loc%cols;

                std::cerr<<"loc "<<loc<<"("<<x<<","<<y<<")"<<" is not reachable from goal "<<goal_loc<<"("<<gy<<","<<gx<<")"<<std::endl;
                std::cerr<<map[loc]<<" "<<map[goal_loc]<<std::endl;
                exit(1);
            }
        }
        
        if (h_vals[loc]>=0) {
            h = h_vals[loc];
        }

        // std::cout << "id ="<< this->id <<", loc =" << loc <<", goal = "<< this->goal_loc << ", h = "<< h <<", expand node = "<< this->all_states.size() <<std::endl;
        return h;
    }

    float get_cost_move(int pst, int ped){
        if (ped-pst==1) {
            // east
            return map_weights[pst*5+0];
        } else if (ped-pst==cols) {
            // south
            return map_weights[pst*5+1];
        } else if (ped-pst==-1) {
            // west
            return map_weights[pst*5+2];
        } else if (ped-pst==-cols) {
            // north
            return map_weights[pst*5+3];
        } else if (ped-pst==0) {
            // stay
            return map_weights[pst*5+4]; // means no move is needed.
        }
        else {
            std::cerr<<"invalid move: "<<pst<<" "<<ped<<std::endl;
            exit(-1);
        }
    }

    bool astar(int loc) {
        bool ret=false;

        while (!open_list.empty()) {
            State *cur = open_list.top();

            // record the shortest path to goal
            h_vals[cur->loc]=cur->g;

            if (cur->loc==loc) {
                ret=true;
                break;
            }

            // we cannot pop before the if line because we need to expand its neighbors
            open_list.pop();
            cur->closed=true;

            int r=cur->loc/cols;
            int c=cur->loc%cols;

            for (int i=0;i<4;++i) {
                int nr=r+dy[i];
                int nc=c+dx[i];
                int nloc=nr*cols+nc;

                if (nr>=0 && nr<rows && nc>=0 && nc<cols && map[nloc]!=1) {

                    // NOTE: we need to get reverse idx and cost
                    int ridx=(i+2)%4;
                    float rcost=map_weights[nloc*5+ridx];
                    // std::cout<<nloc<<"->"<<cur->loc<<" rcost "<<rcost<<std::endl;
                    float g=cur->g+rcost;
                    float h=get_h(nr,nc);

                    State *ns = new State(nloc, g, h);
                    auto it = all_states.find(ns);
                    if (it==all_states.end()) {
                        if (h_vals[nloc]>=0) {
                            std::cout<<nloc<<" sss "<<h_vals[nloc]<<std::endl;
                        }
                        ns->closed = false;
                        ns->open_handle=open_list.push(ns);
                        all_states.insert(ns);
                    } else {
                        auto & old_state=*it;
                        if (ns->g < old_state->g) {
                            old_state->copy(ns);
                            if (old_state->closed) {
                                // reopen: should no happen if we use consistent heuristic, but there might be some numerical issue
                                // let's just not reopen.
                                // old_state->closed=false;
                                // open_list.update(old_state->open_handle);
                                // std::cerr<<"diff: "<<old_state->g<<" "<<ns->g<<std::endl;
                                // std::cerr<<"reopen"<<std::endl;
                                // exit(1);
                            } else {
                                // increase because we have a better new state
                                open_list.update(old_state->open_handle);
                            }
                        }
                        delete ns;
                    }
                }
            }
        }

        return ret;

    }

};


class HeuristicTableV2 {
// private:
public:
    std::vector<float> map_weights;
    const std::vector<int> & map;

    bool update_late=false;

    int rows;
    int cols;
    // const std::vector<float> & map_weights;
    

    float min_map_weight;

    std::vector<LazyBDHeuristicTable> heuristic_tables;

    HeuristicTableV2(int num_of_agents, const std::vector<int> & _map, int _rows, int _cols, const std::vector<float> _map_weights, float _min_map_weight=-1):
        map(_map), rows(_rows), cols(_cols), map_weights(_map_weights), min_map_weight(_min_map_weight) {
        // check map weights' size
        if (map_weights.size()!=_rows*_cols*5) {
            std::cerr<<"map weights' size is not equal to rows*cols*5 (r,d,l,u,wait)"<<std::endl;
            exit(1);
        }

        if (min_map_weight<0) {
            min_map_weight=max_heuristic_val;
            for (size_t i=0;i<map_weights.size();++i) {
                if (map_weights[i]<min_map_weight) {
                    min_map_weight=map_weights[i];
                }
            }
        }

        for (int i=0;i<num_of_agents;++i) {
            heuristic_tables.emplace_back(i, map, rows, cols, map_weights, min_map_weight);
        }
    }

    void update_start_and_goal(int agent_idx, int start_loc, int goal_loc) {
        heuristic_tables[agent_idx].update_start_and_goal(start_loc, goal_loc);
    }

    float get(int agent_idx, int loc) {
        return heuristic_tables[agent_idx].get(loc);
    }

    float get_cost_move(int agent_idx, int pst, int ped) {
        return heuristic_tables[agent_idx].get_cost_move(pst, ped);
    }

    void update_weights(const std::vector<float> _map_weights, const std::vector<::State>& curr_states){
        map_weights = _map_weights;
        min_map_weight = max_heuristic_val;
        for (size_t i=0;i<map_weights.size();++i) {
            if (map_weights[i]<min_map_weight) {
                min_map_weight=map_weights[i];
            }
        }
        // std::cout << "new min weight =" <<min_map_weight<<std::endl;

        for (int i=0; i<heuristic_tables.size(); ++i){
            if (this->update_late){
                heuristic_tables[i].is_newest_weights = false;
                heuristic_tables[i].update_newest_weights(_map_weights, min_map_weight);
            } else {
                heuristic_tables[i].update_start(curr_states[i].location);
                heuristic_tables[i].update_weights_at_once(_map_weights, min_map_weight);
            }
        }
    }

};

}; // namespace HT_v2
