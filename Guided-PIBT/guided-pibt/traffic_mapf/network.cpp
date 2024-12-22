#include "network.hpp"
#include <memory>

namespace TrafficMAPF{
    NetworkConfig network_config = NetworkConfig();
    std::shared_ptr<Network> net_ptr = std::make_shared<LinearNetwork>();
    // torch::Tensor test_tensor= torch::rand({2, 3});
};