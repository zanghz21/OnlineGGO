#ifndef SEARCH_NODE_H
#define SEARCH_NODE_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <limits> 
#include <cmath>

namespace TrafficMAPF{
struct s_node
{
    int label = 0;
    int id = -1; //also location, -1 indicated not generated yet.
    double g = 0;
    int h = 0;
    int op_flow = 0;
    double nn_cost = 0.0;
    double all_vertex_flow = 0;
    bool closed = false;
    int depth = 0;
    double tie_breaker = 0;
    s_node* parent = nullptr;

    unsigned int priority;


    s_node(int id, double g, int h, int op_flow, int depth) : id(id), g(g), h(h), op_flow(op_flow),depth(depth) {};
    s_node() = default;

    double get_f() const {return g + h * 1.0; }
    bool is_closed() const { return closed; }
    void close() { closed = true; }
    int get_op_flow() const { return op_flow; }
    double get_all_vertex_flow() const { return all_vertex_flow; }
    void set_all_flow(int op_flow,  double all_vertex_flow){
        this->op_flow = op_flow;
        this->all_vertex_flow = all_vertex_flow;
    };
    void set_nn_cost(double nn_cost){
        this->nn_cost = nn_cost;
    }
    double get_g() const { return g; }
    int get_h() const { return h; }
    unsigned int get_priority() const { return priority; }
    void set_priority(unsigned int p) { priority = p; }

    void reset(){
        label = 0;
        id = -1;
        g = 0.0;
        h = 0;
        op_flow = 0;
        all_vertex_flow = 0.0;
        closed = false;
        depth = 0;
        parent = nullptr;
        tie_breaker = 0;

    }
    /* data */
};

struct equal_search_node
{
    inline bool operator()(const s_node& lhs, const s_node& rhs) const
    {
        return lhs.id == rhs.id && lhs.get_op_flow() == rhs.get_op_flow() && lhs.get_all_vertex_flow() == rhs.get_all_vertex_flow() && lhs.get_g() == rhs.get_g();
    }
};


struct re_f{
    inline bool operator()(const s_node& lhs, const s_node& rhs) const
    {
        return lhs.get_f() < rhs.get_f();
    }
};

struct re_jam{
    inline bool operator()(const s_node& lhs, const s_node& rhs) const
    {
        if (lhs.get_f() != rhs.get_f())
            return false;
        
        if (lhs.get_op_flow() == rhs.get_op_flow())
        {
                if(lhs.get_all_vertex_flow()+lhs.get_f() == rhs.get_all_vertex_flow() + rhs.get_f()){
                    return  lhs.get_g() > rhs.get_g();   
                }
                else
                return lhs.get_all_vertex_flow()+lhs.get_f() < rhs.get_all_vertex_flow() + rhs.get_f();
        }
        else
            return lhs.get_op_flow() < rhs.get_op_flow();    }
};

struct cmp_less_f
{
    inline bool operator()(const s_node& lhs, const s_node& rhs) const
    {

        if (lhs.get_f() == rhs.get_f()){
                if (lhs.get_g() == rhs.get_g())
                    return rand() % 2;
                else
                    return lhs.get_g() > rhs.get_g();
        }
        else
            return lhs.get_f() < rhs.get_f();

    }
};

struct re_of{
    inline bool operator()(const s_node& lhs, const s_node& rhs) const
    {
        // std::cout <<"lhs = "<<lhs.id<< ", " << lhs.get_op_flow() <<", ";
        // std::cout <<std::setprecision(std::numeric_limits<double>::max_digits10) << lhs.get_f() <<std::endl;
        // std::cout << "rhs = "<<rhs.id<< ", " << rhs.get_op_flow() <<", ";
        // std::cout <<std::setprecision(std::numeric_limits<double>::max_digits10) << rhs.get_f() <<std::endl;
        if (lhs.get_op_flow() == rhs.get_op_flow())
        {   
            if (OBJECTIVE == OBJ::NN){
                return std::round(100*lhs.get_f()) < std::round(100*rhs.get_f());
            } else{
                return lhs.get_f() < rhs.get_f();
            }   
        }
        else
            return lhs.get_op_flow() < rhs.get_op_flow();

    }
};

struct cmp_less_of
{
    inline bool operator()(const s_node& lhs, const s_node& rhs) const
    {

        if (lhs.get_op_flow() == rhs.get_op_flow())
        {
            if (lhs.get_f() == rhs.get_f()){
                #if OBJECTIVE >=9

                    if (lhs.tie_breaker < rhs.tie_breaker)
                        return true;
                    else if (lhs.tie_breaker > rhs.tie_breaker)
                        return false;
                #endif
                if (lhs.get_g() == rhs.get_g())
                    return rand() % 2;
                else
                    return lhs.get_g() > rhs.get_g();
            }
            else
                return lhs.get_f() < rhs.get_f();
        }
        else
            return lhs.get_op_flow() < rhs.get_op_flow();

    }
};

struct cmp_less_jam
{
    inline bool operator()(const s_node& lhs, const s_node& rhs) const
    {
        #if OBJECTIVE >= 6
            if (lhs.tie_breaker < rhs.tie_breaker)
                return true;
            else if (lhs.tie_breaker > rhs.tie_breaker)
                return false;
        #endif

        if (lhs.get_op_flow() == rhs.get_op_flow())
        {
            if (lhs.get_all_vertex_flow() +lhs.get_f()  == rhs.get_all_vertex_flow() + rhs.get_f())
            {
                if (lhs.get_all_vertex_flow() + lhs.get_g() == rhs.get_all_vertex_flow() + rhs.get_g()){
                    if (lhs.get_g() == rhs.get_g())
                        return rand() % 2;
                    else
                        return lhs.get_g() > rhs.get_g();

                }
                return lhs.get_all_vertex_flow() + lhs.get_g() < rhs.get_all_vertex_flow() + rhs.get_g();    
            }
            else
                return lhs.get_all_vertex_flow()+lhs.get_f() < rhs.get_all_vertex_flow() + rhs.get_f();
        }
        else
            return lhs.get_op_flow() < rhs.get_op_flow();

    }
};

}
#endif