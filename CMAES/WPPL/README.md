# Windowed Parallel PIBT LNS

This repo is forked from the implementation of the lifelong MAPF algorithm **Windowed Parallel PIBT LNS (WPPL)**, described in the paper **[Scaling Lifelong Multi-Agent Path Finding to More Realistic Settings: Research Challenges and Opportunities](https://arxiv.org/abs/2404.16162)**, accepted to [SoCS 2024](https://socs24.search-conference.org/). The original repo could be found at [here](https://github.com/DiligentPanda/MAPF-LRR2023).

Mainly, this forked version has several key differences from the original version:

1. The action model of the robots/agents is four-way-moving (i.e. moving up, down, left, right), instead of rotating clockwise/counter-clockwise and moving forward.

2. This version supports the warehouse map, in which there are endpoints/workstations between which the robots move. See section 2 of [this paper](https://arxiv.org/abs/2305.06436) for more details.

3. This repo is mostly used as a *C++ subroutine* by *Python* projects. Therefore compiling this projects requires [pybind11](https://pybind11.readthedocs.io/en/stable/). One example usage of this repo can be found at the [guidance graph optimization repo](https://github.com/lunjohnzhang/ggo_public).


## Compile

### Denpendencies

- [cmake >= 3.16](https://cmake.org/)
- [libboost >= 1.49.0](https://www.boost.org/)
- Python3 and [pybind11](https://pybind11.readthedocs.io/en/stable/)

Install dependencies on Ubuntu or Debian Linux:
```shell
sudo apt-get update
sudo apt-get install build-essential libboost-all-dev python3-dev python3-pybind11 
```

[Homebrew](https://brew.sh/) is recomanded for installing dependencies on Mac OS.

### Compiling

Using cmake: 
```shell
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release
make -j
```

## Run

```bash
python run_py_driver.py
```