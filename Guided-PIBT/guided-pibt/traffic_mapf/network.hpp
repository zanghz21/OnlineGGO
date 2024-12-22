#pragma once
#include <memory>
#include <vector>
#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>
// #include <torch/torch.h>
#include <MiniDNN.h>
#include <Eigen/Dense>

namespace TrafficMAPF
{   
    // struct torch_mlp_net : torch::nn::Module {
    //     torch_mlp_net(int input_size, int hidden_size, int output_size): 
    //         input_size(input_size),
    //         hidden_size(hidden_size), 
    //         output_size(output_size), 
    //         fc1(input_size, hidden_size), 
    //         fc2(hidden_size, output_size) {
    //         register_module("fc1", fc1);
    //         register_module("fc2", fc2);
    //         params_num = getTotalParams();
    //     }

    //     torch::Tensor forward(torch::Tensor x) {
    //         // std::cout << "run weights = " << this->fc1->weight.slice(0, 0, 1) << std::endl;
    //         x = x.view({-1, input_size});
    //         x = this->fc1->forward(x);
    //         x = torch::relu(x);
    //         x = this->fc2->forward(x);
    //         return x;
    //     }

    //     int getTotalParams() {
    //         int totalParams = 0;
    //         for (const auto& parameter : parameters()) {
    //             totalParams += parameter.numel(); // Count the number of elements in each parameter tensor
    //         }
    //         std::cout << "Total params ="<<totalParams<<std::endl;
    //         return totalParams;
    //     }

    //     void setWeights(const std::vector<double>& all_weights_) {

    //         std::vector<float> all_weights;
    //         for (const auto& ele : all_weights_) {
    //             all_weights.push_back(static_cast<float>(ele));
    //         }
            
    //         auto weights1 = torch::from_blob(all_weights.data(), {hidden_size, input_size}, torch::kFloat32);
    //         auto bias1 = torch::from_blob(all_weights.data()+weights1.numel(), {hidden_size}, torch::kFloat32);
    //         auto weights2 = torch::from_blob(all_weights.data()+weights1.numel()+bias1.numel(), {output_size, hidden_size}, torch::kFloat32);
    //         auto bias2 = torch::from_blob(all_weights.data()+weights1.numel()+bias1.numel()+weights2.numel(), {output_size}, torch::kFloat32);

    //         // Set weights for layers
    //         this->fc1->weight.set_data(weights1.clone());
    //         this->fc1->bias.set_data(bias1.clone());
    //         this->fc2->weight.set_data(weights2.clone());
    //         this->fc2->bias.set_data(bias2.clone());

    //         this->fc1->eval();
    //         this->fc2->eval();
    //     }

    //     int input_size;
    //     int hidden_size;
    //     int output_size;
    //     int params_num;
    //     torch::nn::Linear fc1;
    //     torch::nn::Linear fc2;
    // };
    
    class NetworkConfig {
        public:
            double default_obst_flow = 0.0;
            bool learn_obst_flow = false;
            int WIN_R = 2;
            int hidden_size = 50;
            const int map_hidden_size = 10;
            int output_size = 4;
            bool has_map = false;
            bool has_path = false;
            bool has_previous = true;
            bool use_all_flow = false;
            bool use_cached_nn = false;
            bool use_rotate_input = false;
            std::string input_type = "flow";
    };
    extern NetworkConfig network_config;
   
    
    class Network {
        protected:
            int input_channel;
            int input_size;
            int output_size;
            int map_input_size;
            bool has_map;
            int WIN_R;
            std::vector<double> params;

            double clip_result(double result){
                result = std::min(1000.0, result);
                result = std::max(0.0, result);
                return std::round(result*100)/100.0;
            }
        public:
            int params_num=-1;
            Network(){
                this->WIN_R = network_config.WIN_R;
                this->has_map = network_config.has_map;
                this->has_map? this->map_input_size = (2*WIN_R+1)*(2*WIN_R+1): this->map_input_size = 0;

                if (network_config.input_type == "flow"){
                    this->input_channel = 4;
                } else if (network_config.input_type == "traf_and_goal"){
                    this->input_channel = 6;
                } else if (network_config.input_type == "flow_and_traf_and_goal") {
                    this->input_channel = 10;
                } else if (network_config.input_type == "minimum"){
                    this->input_channel = 1;
                } else {
                    std::cout << "network input_type ["<<network_config.input_type<<"] is not supported!"<<std::endl;
                    exit(-1);
                }
                
                this->input_size = this->input_channel*(2*this->WIN_R+1)*(2*this->WIN_R+1);
                this->output_size = network_config.output_size;
            };
            virtual std::vector<double> forward(std::vector<double>& input_vec, std::vector<double>& map_input){
                std::cout<<"in Network::forward"<<std::endl;
                exit(-1);
                return std::vector<double>();
            }
            virtual double forward(int output_dim, std::vector<double>& input_vec, std::vector<double>& map_input){
                std::cout<<"in Network::forward, return double"<<std::endl;
                exit(-1);
                return 0.0;
            }
            virtual void set_params(std::vector<double> new_params){
                std::cout<<"in Network::set_params"<<std::endl;
                exit(-1);
            }
            std::vector<double> get_params() const {
                return this->params;
            }
    };

    class LinearNetwork: public Network {
        public:
            LinearNetwork(): Network(){
                this->params_num = (this->input_size + this->map_input_size)* this->output_size;
                this->params.clear();
                this->params.resize(this->params_num, 0.);
                // use Eigen will be slower
            }
            std::vector<double> forward(std::vector<double>& input_vec, std::vector<double>& map_input) override {
                // std::cout << "in LinearNetwork::forward" <<std::endl;
                // can also be used for traf+goal observation
                if (input_vec.size() != this->input_size){
                    std::cout << "Network is not compatible with input observation: input size = "<<input_vec.size();
                    std::cout<<", network input size ="<< this->input_size<<std::endl; 
                    exit(1);
                }

                std::vector<double> result(this->output_size, 0.0);
                for (int i=0; i<this->output_size; ++i){
                    int start_id = i*(this->input_size+this->map_input_size);
                    for (int j=0; j<this->input_size; ++j){
                        result[i] += input_vec[j] * this->params[start_id+j];
                    }
                    for (int j=0; j<this->map_input_size; ++j){
                        result[i] += map_input[j] * this->params[start_id+this->input_size+j];
                    }
                    result[i] = this->clip_result(result[i]);
                }
                return result;
            }
            double forward(int output_dim, std::vector<double>& input_vec, std::vector<double>& map_input) override {
                // std::cout << "in LinearNetwork::forward" <<std::endl;
                if (input_vec.size() != this->input_size){
                    std::cout << "Network is not compatible with input observation: input size = "<<input_vec.size();
                    std::cout<<", network input size ="<< this->input_size<<std::endl; 
                    exit(1);
                }
                double result = 0.0;
                int start_id = output_dim*(this->input_size+this->map_input_size);
                for (int j=0; j<this->input_size; ++j){
                    result += input_vec[j] * this->params[start_id+j];
                }
                for (int j=0; j<this->map_input_size; ++j){
                    result += map_input[j] * this->params[start_id+this->input_size+j];
                }
                result = this->clip_result(result);
                return result;
            }

            void set_params(std::vector<double> new_params) override {
                assert(new_params.size() == this->params_num);
                this->params = new_params;
            }
    };

    class QuadraticNetwork: public Network {
        int common_edges;
        public: 
            QuadraticNetwork(): Network(){
                this->common_edges = 4*this->WIN_R*(2*this->WIN_R+1);
                this->params_num = (this->input_size + this->common_edges 
                                    + this->map_input_size + this->map_input_size*4) * this->output_size;
                if (network_config.input_type == "flow_and_traf_and_goal"){
                    this->params_num += this->common_edges * this->output_size;
                }
                this->params.clear();
                this->params.resize(this->params_num, 1.0);
            }
            std::vector<double> forward(std::vector<double>& input_vec, std::vector<double>& map_input) override {
                // double result = 0.0;
                // for traf+goal observation, we need to ensure that the first 4 dim to be RDLU, and goal obs should serve as the 6 dim
                // for flow+traf+goal observation, flow-4d, traf-5d, goal-1d
                std::vector<double> result(this->output_size, 0.0);
                int params_id = 0;

                // linear terms
                for (int i=0; i<this->output_size; ++i){
                    for (int j=0; j<this->input_size; ++j){
                        result[i] += input_vec[j] * this->params[params_id];
                        params_id++;
                    }
                    for (int j=0; j<this->map_input_size; ++j){
                        result[i] += map_input[j] * this->params[params_id];
                        params_id++;
                    }
                }

                // quad terms with maps
                for (int i=0; i<this->output_size; ++i){
                    for (int j=0; j<this->map_input_size; ++j){
                        for(int k=0; k<this->input_channel; ++k){
                            result[i] += map_input[j] * input_vec[this->input_channel*j+k] * this->params[params_id];
                            params_id++;
                        }
                    }
                }

                // quad terms for head-on-head edge usage
                for (int i=0; i<this->output_size; ++i){
                    for (int id=0; id<(2*this->WIN_R+1)*(2*this->WIN_R+1); ++id){
                        if (id % (2*this->WIN_R+1) != (2*this->WIN_R)){
                            int right_id = id * this->input_channel;
                            int left_id = right_id + this->input_channel+2;
                            result[i] += this->params[params_id] * (input_vec[right_id]*input_vec[left_id]);
                            params_id += 1;
                            if (network_config.input_type == "flow_and_traf_and_goal"){
                                int right_id2 = id * this->input_channel + 4;
                                int left_id2 = right_id + this->input_channel + 6;
                                result[i] += this->params[params_id] * (input_vec[right_id2]*input_vec[left_id2]);
                                params_id += 1;
                            }
                        }
                        if (id < (2*this->WIN_R)*(2*this->WIN_R+1)){
                            int down_id = id*this->input_channel +1;
                            int up_id = down_id + (2* this->WIN_R + 1)*this->input_channel + 2;
                            result[i] += this->params[params_id] *(input_vec[down_id]*input_vec[up_id]);
                            params_id += 1;
                            if (network_config.input_type == "flow_and_traf_and_goal"){
                                int down_id2 = id*this->input_channel + 5;
                                int up_id2 = down_id2 + (2* this->WIN_R + 1)*this->input_channel + 2;
                                result[i] += this->params[params_id] *(input_vec[down_id2]*input_vec[up_id2]);
                                params_id += 1;
                            }
                        }
                    }
                    result[i] = this->clip_result(result[i]);
                }
                // std::cout << "after all terms:"<< params_id<<std::endl;
                if (params_id != this->params.size()){
                    std::cout << "QuadNet: potential bug exist in forward"<<std::endl;
                    exit(-1);
                }
                return result;
            }

            double forward(int output_dim, std::vector<double>& input_vec, std::vector<double>& map_input) override {
                double result = 0.0;
                
                int start_id = output_dim*(this->input_size+this->map_input_size);
                for (int j=0; j<this->input_size; ++j){
                    result += input_vec[j] * this->params[start_id+j];
                }
                for (int j=0; j<this->map_input_size; ++j){
                    result += map_input[j] * this->params[start_id+this->input_size];
                }
                // std::cout << "after linear term, params id ="<< params_id<<std::endl;
                start_id = this->output_size*(this->input_size+this->map_input_size) + output_dim*this->map_input_size*4;
                for (int j=0; j<this->map_input_size; ++j){
                    for(int k=0; k<this->input_channel; ++k){
                        result += map_input[j] * input_vec[this->input_channel*j+k] * this->params[start_id+this->input_channel*j+k];
                    }
                }
                start_id = (this->output_size*(this->input_size+this->map_input_size + this->map_input_size*4)) + output_dim*this->common_edges;
                int params_id = start_id;
                for (int id=0; id<(2*this->WIN_R+1)*(2*this->WIN_R+1); ++id){
                    // std::cout << "id =" <<id <<std::endl;
                    if (id % (2*this->WIN_R+1) != (2*this->WIN_R)){
                        int right_id = id * this->input_channel;
                        int left_id = right_id + 6;
                        // std::cout << "left, right ="<<right_id<<", "<<left_id<<std::endl;
                        
                        // if(this->params[params_id]!=0){
                            // std::cout << "id =" <<id <<", SUB_COST =" <<input_vec[right_id]<<", "<<input_vec[left_id]<<std::endl;
                        // }
                        result += this->params[params_id] * (input_vec[right_id]*input_vec[left_id]);
                        params_id += 1;
                    }
                    if (id < (2*this->WIN_R)*(2*this->WIN_R+1)){
                        int down_id = id*this->input_channel +1;
                        int up_id = down_id + (2* this->WIN_R + 1)*this->input_channel + 2;
                        // if(this->params[params_id]!=0){
                            // std::cout << "id =" <<id <<", SUB_COST =" <<this->params[params_id] *(input_vec[down_id]*input_vec[up_id])<<std::endl;
                        // }
                        result += this->params[params_id] *(input_vec[down_id]*input_vec[up_id]);
                        params_id += 1;
                    }
                }
                result = this->clip_result(result);
                // std::cout << "after all terms:"<< params_id<<std::endl;
                return result;
            }
            void set_params(std::vector<double> new_params) override {
                this->params = new_params;
            }
    };

    class MinimumNetwork: public Network{
        public:
            MinimumNetwork(): Network(){
                this->params_num = 48; // 4 direction; each dir, 4 v, 4 e-linear, 4 e-quad
                this->input_size = 20;
                this->params.clear();
                this->params.resize(this->params_num, 0);
                if (network_config.input_type != "minimum"){
                    std::cout<<"MinimumNetwork only support [minimum] input, not ["<<network_config.input_type<<"]!"<<std::endl;
                    exit(-1);
                }
            }
            std::vector<double> forward(std::vector<double>& input_vec, std::vector<double>&map_input) override{
                if (input_vec.size() != this->input_size){
                    std::cout << "Network is not compatible with input observation: input size = "<<input_vec.size();
                    std::cout<<", network input size ="<< this->input_size<<std::endl; 
                    exit(1);
                }
                std::vector<double> result(this->output_size, 0.0);
                int p_id = 0;

                // std::cout << "input:" <<std::endl;
                // for (auto v: input_vec){
                //     std::cout << v <<", ";
                // }
                // std::cout <<std::endl;

                // std::cout << "param:" <<std::endl;
                // for (auto p: this->params){
                //     std::cout << p <<", ";
                // }
                // std::cout << std::endl;

                int edges_ids[4] = {4*1+2, 4*2+3, 4*3+0, 4*4+1}; //R, D, L, U
                // std::cout << "results:" <<std::endl; 
                for (int i=0; i<this->output_size; ++i){
                    // v usage
                    for (int j=0; j<4; ++j){
                        result[i] += this->params[p_id] * input_vec[4*(i+1)+j];
                        p_id++;
                    }
                    // e usage
                    for (int j=0; j<4; ++j){
                        result[i] += this->params[p_id] * input_vec[edges_ids[j]];
                        p_id++;
                    }
                    for (int j=0; j<4; ++j){
                        result[i] += this->params[p_id] * (input_vec[j]*input_vec[edges_ids[j]]);
                        p_id++;
                    }


                    // std::cout << result[i] << ", ";
                    result[i] = this->clip_result(result[i]);
                }
                // std::cout << std::endl;
                if (p_id != this->params.size()){
                    std::cout << "must bug"<<std::endl;
                    exit(1);
                }
                return result;
            }
            void set_params(std::vector<double> new_params) override {
                assert(new_params.size() == this->params_num);
                this->params = new_params;
            }
    };
    
    class TorchMLPNetwork: public Network{
        // private:
        //     int hidden_size = 20;
        //     torch_mlp_net net;
        // public:
        //     TorchMLPNetwork(): 
        //         Network(), 
        //         net(this->input_size, this->hidden_size, this->output_size){
        //             std::cout << "MLPNetwork init" <<std::endl;
        //             this->params_num = this->net.params_num;
        //         }
        //     std::vector<double> forward(std::vector<double> input_vec, std::vector<double> map_input=std::vector<double>()) override {
        //         if (input_vec.size() != this->input_size){
        //             std::cout << "Network is not compatible with input observation: input size = "<<input_vec.size();
        //             std::cout<<", network input size ="<< this->input_size<<std::endl; 
        //             exit(1);
        //         }
        //         std::vector<float> input_vec_fl;
        //         for (const auto& ele : input_vec) {
        //             input_vec_fl.push_back((static_cast<float>(ele)));
        //         }
        //         auto input_tensor = torch::from_blob(input_vec_fl.data(), {1, this->input_size}, torch::kFloat32);
        //         auto res_tensor = this->net.forward(input_tensor);
                
        //         int64_t numel = res_tensor.numel();
        //         const float* data_ptr = res_tensor.data_ptr<float>();
        //         std::vector<float> float_result(numel);
        //         std::memcpy(float_result.data(), data_ptr, numel * sizeof(float));
        //         std::vector<double> result;
        //         for(int i=0; i<this->output_size; ++i){
        //             result.push_back(static_cast<float>(float_result[i]));
        //         }
        //         return result;
        //     }

        //     void set_params(std::vector<double> new_params) override {
        //         if (new_params.size() != this->params_num){
        //             std::cout << "error params num!" <<std::endl;
        //             exit(1);
        //         }
        //         this->net.setWeights(new_params);
        //         this->params = new_params;
        //     }
    };

    class MyMLPNetwork: public Network{
        private:
            std::vector<std::vector<double>> fc1_w;
            std::vector<double> fc1_b;
            std::vector<std::vector<double>> fc2_w;
            std::vector<double> fc2_b;
            int hidden_size;
            int map_hidden_size;
        public:
            MyMLPNetwork(): Network(), 
            hidden_size(network_config.hidden_size)
            {   
                // std::cout << "in MyMLPNetwork::init" <<std::endl;
                this->has_map? this->map_hidden_size=network_config.map_hidden_size: this->map_hidden_size=0;
                fc1_w = std::vector<std::vector<double>>(hidden_size+map_hidden_size, std::vector<double>(input_size+map_input_size, 0.0));
                fc1_b = std::vector<double>(hidden_size+map_hidden_size, 0.0);
                fc2_w = std::vector<std::vector<double>>(this->output_size, std::vector<double>(hidden_size+map_hidden_size, 0.0));
                fc2_b = std::vector<double>(this->output_size, 0.0);
                this->params_num = (this->input_size+this->map_input_size) * (this->hidden_size+this->map_hidden_size) 
                    + (this->hidden_size+this->map_hidden_size) 
                    + (this->hidden_size+this->map_hidden_size) * this->output_size
                    + this->output_size;
                // std::cout << "end MyMLPNetwork::init" <<std::endl;
            }
            std::vector<double> forward(std::vector<double>& input_vec, std::vector<double>& map_input) override {
                std::vector<double> fc1_res(this->hidden_size+this->map_hidden_size, 0.0);
                // std::cout << "in MyMLPNetwork::forward, begin()" <<std::endl;
                // fc1
                for(int i=0; i<this->hidden_size+this->map_hidden_size; ++i){
                    for(int j=0; j<this->input_size; ++j){
                        fc1_res[i] += this->fc1_w[i][j] * input_vec[j];
                    }
                    for(int j=0; j<this->map_input_size; ++j){
                        fc1_res[i] += this->fc1_w[i][this->input_size+j] * map_input[j];
                    }
                    fc1_res[i] += this->fc1_b[i];
                }
                // std::cout << "after fc1"<<std::endl;
                // relu
                for(int i=0; i<this->hidden_size+this->map_hidden_size; ++i){
                    if (fc1_res[i] < 0.0){
                        fc1_res[i] = 0.0;
                    }
                }
                // std::cout << "after relu"<<std::endl;
                // fc2
                std::vector<double> result(this->output_size, 0.0);
                for (int i=0; i<this->output_size; ++i){
                    for(int j=0; j<this->hidden_size+this->map_hidden_size; ++j){
                        result[i] += this->fc2_w[i][j] * fc1_res[j];
                    }
                    result[i] += this->fc2_b[i];
                    result[i] = this->clip_result(result[i]);
                }
                // std::cout << "after fc2" <<std::endl;
                // std::cout << "in MyMLPNetwork::forward, end()" <<std::endl;
                return result;
            }

            double forward(int output_dim, std::vector<double>& input_vec, std::vector<double>& map_input) override {
                std::vector<double> fc1_res(this->hidden_size+this->map_hidden_size, 0.0);
                // std::cout << "in MyMLPNetwork::forward, begin()" <<std::endl;
                // fc1
                for(int i=0; i<this->hidden_size+this->map_hidden_size; ++i){
                    for(int j=0; j<this->input_size; ++j){
                        fc1_res[i] += this->fc1_w[i][j] * input_vec[j];
                    }
                    for(int j=0; j<this->map_input_size; ++j){
                        fc1_res[i] += this->fc1_w[i][this->input_size+j] * map_input[j];
                    }
                    fc1_res[i] += this->fc1_b[i];
                }
                // std::cout << "after fc1"<<std::endl;
                // relu
                for(int i=0; i<this->hidden_size+this->map_hidden_size; ++i){
                    if (fc1_res[i] < 0.0){
                        fc1_res[i] = 0.0;
                    }
                }
                double result;
                for(int j=0; j<this->hidden_size+this->map_hidden_size; ++j){
                    result += this->fc2_w[output_dim][j] * fc1_res[j];
                }
                result += this->fc2_b[output_dim];
                result = this->clip_result(result);
                return result;
            }

            void set_params(std::vector<double> new_params) override {
                // std::cout << "params num =" <<this->params_num << ", set params num ="<< new_params<<std::endl;
                int start_id =0;
                for (int i=0; i<this->hidden_size+this->map_hidden_size; ++i){
                    for (int j=0; j<this->input_size+this->map_input_size; ++j){
                        this->fc1_w[i][j] = new_params[start_id];
                        start_id++;
                    }
                }
                for (int i=0; i<this->hidden_size+this->map_hidden_size; ++i){
                    this->fc1_b[i]=new_params[start_id];
                    start_id++;
                }
                for (int i=0; i<this->output_size; ++i){
                    for (int j=0; j<this->hidden_size+this->map_hidden_size; ++j){
                        this->fc2_w[i][j]=new_params[start_id];
                        start_id++;
                    }
                }
                for (int i=0; i<this->output_size; ++i){
                    this->fc2_b[i] = new_params[start_id];
                    start_id++;
                }
                // std::cout<<"successfully set weights"<<std::endl;
                if(start_id != new_params.size()){
                    std::cout<<"MyMLPNetwork::set_params, there must be bugs"<<std::endl;
                    exit(-1);
                }
            }
    };
    
    class MiniDNNMLPNetwork: public Network{
        private:
        int hidden_size;
        int map_hidden_size;
        MiniDNN::Network net;
        public:
            MiniDNNMLPNetwork(): Network(), 
            hidden_size(network_config.hidden_size)
            {
                this->has_map? this->map_hidden_size=network_config.map_hidden_size: this->map_hidden_size=0;
                this->params_num = (this->input_size+this->map_input_size) * (this->hidden_size+this->map_hidden_size) 
                    + (this->hidden_size+this->map_hidden_size) 
                    + (this->hidden_size+this->map_hidden_size) * this->output_size
                    + this->output_size;
                MiniDNN::Layer* layer1 = new MiniDNN::FullyConnected<MiniDNN::ReLU>(
                    (this->input_size + this->map_input_size), (this->hidden_size+this->map_hidden_size)
                );
                MiniDNN::Layer* layer2 = new MiniDNN::FullyConnected<MiniDNN::Identity>(
                    (this->hidden_size+this->map_hidden_size), this->output_size
                );
                net.add_layer(layer1);
                net.add_layer(layer2);
                net.init(0, 0.01, 123);
            }
            double forward(int output_dim, std::vector<double>& input_vec, std::vector<double>& map_input) override{
                Eigen::Map<Eigen::VectorXd> input_eigen_vec(input_vec.data(), input_vec.size());
                Eigen::MatrixXd input_mat = input_eigen_vec;
                
                if (!map_input.empty() && this->map_input_size > 0) {
                    Eigen::Map<Eigen::VectorXd> map_input_eigen_vec(map_input.data(), map_input.size());
                    Eigen::MatrixXd map_input_mat = map_input_eigen_vec;
                    input_mat.conservativeResize(this->input_size+this->map_input_size, 1);
                    input_mat << input_eigen_vec, map_input_mat;
                }
                Eigen::MatrixXd output_mat = net.predict(input_mat);
                return this->clip_result(output_mat(output_dim, 0));
            }

            std::vector<double> forward(std::vector<double>& input_vec, std::vector<double>& map_input) override{
                Eigen::Map<Eigen::VectorXd> input_eigen_vec(input_vec.data(), input_vec.size());
                Eigen::MatrixXd input_mat = input_eigen_vec;

                if (!map_input.empty() && this->map_input_size > 0) {
                    Eigen::Map<Eigen::VectorXd> map_input_eigen_vec(map_input.data(), map_input.size());
                    Eigen::MatrixXd map_input_mat = map_input_eigen_vec;
                    input_mat.conservativeResize(this->input_size+this->map_input_size, 1);
                    input_mat << input_eigen_vec, map_input_mat;
                }
                Eigen::MatrixXd output_mat = net.predict(input_mat);
                
                std::vector<double> result;
                for(int i=0; i<this->output_size; ++i){
                    result.push_back(this->clip_result(output_mat(i, 0)));
                }
                return result;
            }
            void set_params(std::vector<double> new_params) override {
                // std::cout<<"in MiniDNNMLPNetwork::set_params"<<std::endl;
                auto layers = this->net.get_layers();
                int param_start = 0;
                std::vector<std::vector<double>> collected_params;
                for (auto layer: layers){
                    int pn = (layer->get_parameters()).size();
                    std::vector<double> layer_params(new_params.begin()+param_start, new_params.begin()+param_start+pn);
                    collected_params.push_back(layer_params);
                    param_start += pn;
                }
                this->net.set_parameters(collected_params);
            }
    };
    extern std::shared_ptr<Network> net_ptr;
    // extern torch::Tensor test_tensor;
} // namespace TrafficMAPF