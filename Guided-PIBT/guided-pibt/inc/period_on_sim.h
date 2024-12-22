#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include "CompetitionSystem.h"
#include "nlohmann/json.hpp"
#include "dev.h"
#include <signal.h>
#include <climits>
#include <memory>
#include <random>

namespace po = boost::program_options;
using json = nlohmann::json;
namespace py = pybind11;

class period_on_sim {
    public:
        period_on_sim(py::kwargs kwargs);
        std::string warmup();
        std::string update_gg_and_step(std::vector<double> map_weights);
    private:
        std::vector<int> tasks;
        std::vector<int> agents;
        std::unique_ptr<BaseSystem> py_system_ptr;
        MAPFPlanner* planner = nullptr;
        Grid grid;
        int simuTime = 1000;

        int warmup_time;
        int update_gg_interval;

};