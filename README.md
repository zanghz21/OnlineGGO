# Online Guidance Graph Optimization for Lifelong Multi-Agent Path Finding
This repository is the official implementation of Online Guidance Graph Optimization for Lifelong Multi-Agent Path Finding, accepted to AAAI 2025. The repository builds on top of the repositories of [Guidance Graph Optimization for Lifelong Multi-Agent Path Finding](https://github.com/lunjohnzhang/ggo_public), [Scaling Lifelong Multi-Agent Path Finding to More Realistic Settings: Research Challenges and Opportunities](https://github.com/DiligentPanda/MAPF-LRR2023), and [Trafficflow Optimisation for Lifelong Multi-Agent Path Finding](https://github.com/nobodyczcz/Guided-PIBT).

## Installation
This is a hybrid C++/Python project. The simulation environment is written in C++ and the rests are in Python. We use [pybind11](https://pybind11.readthedocs.io/en/stable/) to bind the two languages.

1. **Install Singularity:** All of our code runs in a Singularity container.
   Singularity is a container platform (similar in many ways to Docker). Please
   see the instructions [here](https://sylabs.io/docs/) for installing Singularity.
   As a reference, we use version 3.10.5 on Ubuntu 20.04.

1. **Clone Repo:**
   ```
   git clone git@github.com:zanghz21/OnlineGGO.git
   git submodule init
   git submodule update
   ```

1. **Download Boost:** From the root directory of the project, run the following to download the Boost 1.71, which is required for compiling C++ simulator. You don't have to install it on your system since it will be passed into the container and installed there.

   ```
   wget https://boostorg.jfrog.io/artifactory/main/release/1.71.0/source/boost_1_71_0.tar.gz --no-check-certificate
   ```

1. **Downloading torch cpp library**
    Refering to https://pytorch.org/cppdocs/installing.html. 
    ```
    # under base_dir
    wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
    mkdir test_torch
    mv libtorch-shared-with-deps-latest.zip test_torch
    cd test_torch
    unzip libtorch-shared-with-deps-latest.zip
    cd ..
    ```

1. **Build Singularity container:** Run the provided script to build the container. Note that this need `sudo` permission on your system.
   ```
   bash build_container.sh
   ```
   The script will first build a container as a sandbox, compile the C++ simulator, then convert that to a regular `.sif` Singularity container.

## (Online) Guidance Policy Optimization
### Runing command
**The following code is running under CMAES/ directory**
```bash
# set CONFIG to the path of config
# set SEED to a positive integer. Note that you cannot run 2 experiments with a same seed concurrently. 
# set NUM_WORKERS to a positive integer. better no larger than the number of the physical cores on the machine
# set PROJECT_DIR to the absolute path of the base dir (the parent dir of CMAES/)
bash scripts/run_local.sh $CONFIG $SEED $NUM_WORKERS -p $PROJECT_DIR
```

### Config files

| sortation-static | config                                                             |
|--------------|--------------------------------------------------------------------|
| on+PIBT      | config/competition/online_update/sortation_small_800_traf_task.gin |
| off+PIBT     | config/competition/cnn_iter_update/sortation_small_800.gin         |
| on+GPIBT     | config/traffic_mapf/sortation_small.gin                            |
| [p-on]+GPIBT | config/traffic_mapf/period_online/sortation_small.gin                                                                |
| off+GPIBT    | config/traffic_mapf/offline/sortation_small.gin                                                                   |

| sortation-dynamic | config |
|-------------------|--------|
| on+PIBT           | config/competition/online_update/sortation_small_800_dists_time_traf_task_sigma.gin |
| off+PIBT          | config/competition/cnn_iter_update/sortation_small_800_dists_time_sigma.gin |
| on+GPIBT          | config/traffic_mapf/sortation_small_dist_sigma.gin |
| [p-on]+GPIBT      | config/traffic_mapf/period_online/sortation_small_dist_sigma.gin |
| off+GPIBT         | config/traffic_mapf/offline/sortation_small_dist_sigma.gin |

| warehouse-static | config |
|-------------------|--------|
| on+PIBT           | config/competition/online_update/33x57_wnarrow_600.gin |
| off+PIBT          | config/competition/cnn_iter_update/33x57_wnarrow_600.gin |
| on+GPIBT          | config/traffic_mapf/warehouse_small_narrow.gin |
| [p-on]+GPIBT      | config/traffic_mapf/period_online/wnarrow.gin |
| off+GPIBT         | config/traffic_mapf/offline/warehouse_small_narrow.gin |
| on+GPIBT+LNS      | config/traffic_mapf/with_lns/warehouse_small_narrow.gin |
| off+GPIBT+LNS     | config/traffic_mapf/offline/with_lns/warehouse_small_narrow.gin |
| on+PIBT, m=200    | config/competition/online_update/ablation_m/33x57_wnarrow_600_m200.gin |
| on+PIBT, m=100    | config/competition/online_update/ablation_m/33x57_wnarrow_600_m100.gin |
| on+PIBT, m=50     | config/competition/online_update/ablation_m/33x57_wnarrow_600_m50.gin |
| on+PIBT, m=20     | config/competition/online_update/33x57_wnarrow_600.gin |
| on+PIBT, m=10     | config/competition/online_update/ablation_m/33x57_wnarrow_600_m10.gin |


| warehouse-dynamic | config |
|-------------------|--------|
| on+PIBT, 400 agents | config/competition/online_update/33x57_wnarrow_400_traf_task_gaussian_sigma.gin |
| off+PIBT, 400 agents | config/competition/cnn_iter_update/33x57_wnarrow_400_dists.gin |
| on+PIBT, 600 agents | config/competition/online_update/33x57_wnarrow_600_traf_task_gaussian_sigma.gin |
| off+PIBT, 600 agents | config/competition/cnn_iter_update/33x57_wnarrow_600_dists.gin |
| on+GPIBT          | config/traffic_mapf/warehouse_small_narrow_dist_sigma.gin |
| [p-on]+GPIBT      | config/traffic_mapf/period_online/warehouse_small_narrow_dist_sigma.gin |
| off+GPIBT         | config/traffic_mapf/offline/warehouse_small_narrow_dist_sigma.gin |
| on+GPIBT+LNS      | config/traffic_mapf/with_lns/warehouse_small_narrow_dist.gin |
| off+GPIBT+LNS     | config/traffic_mapf/offline/with_lns/warehouse_small_narrow_dist_sigma.gin |
|-------------------|--------|
| on+PIBT, m=200    | config/competition/online_update/ablation_m/33x57_wnarrow_600_dist_m200.gin |
| on+PIBT, m=100    | config/competition/online_update/ablation_m/33x57_wnarrow_600_dist_m100.gin |
| on+PIBT, m=50     | config/competition/online_update/ablation_m/33x57_wnarrow_600_dist_m50.gin |
| on+PIBT, m=20     | config/competition/online_update/33x57_wnarrow_400_traf_task_gaussian_sigma.gin |
| on+PIBT, m=10     | config/competition/online_update/ablation_m/33x57_wnarrow_600_dist_m10.gin |


| empty-static | config |
|---------------|--------|
| on+PIBT       | config/competition/online_update/32x32_empty_400_task_traf.gin |
| off+PIBT      | config/competition/cnn_iter_update/32x32_empty_400.gin |
| on+GPIBT      | config/traffic_mapf/empty.gin |
| [p-on]+GPIBT  | config/traffic_mapf/period_online/empty.gin |
| off+GPIBT     | config/traffic_mapf/offline/empty.gin |


| empty-dynamic | config |
|---------------|--------|
| on+PIBT       | config/competition/online_update/32x32_empty_400_task+traf_dist.gin |
| off+PIBT      | config/competition/cnn_iter_update/32x32_empty_400_dists.gin |
| on+GPIBT      | config/traffic_mapf/empty_dist.gin |
| [p-on]+GPIBT  | config/traffic_mapf/period_online/empty_dist.gin |
| off+GPIBT     | config/traffic_mapf/offline/empty_dist.gin | 
| on+GPIBT+min  | CMAES/config/traffic_mapf/empty_dist_min.gin |

| random-static | config |
|----------------|--------|
| on+PIBT        | config/competition/online_update/32x32_random_400_traf_task.gin |
| off+PIBT       | config/competition/cnn_iter_update/CMA_ES_PIBT_32x32_random-map_400-agents_iter_update.gin |
| on+GPIBT       | config/traffic_mapf/random.gin |
| [p-on]+GPIBT   | config/traffic_mapf/period_online/random.gin |
| off+GPIBT      | config/traffic_mapf/offline/random.gin |
| on+GPIBT+min   | config/traffic_mapf/random_dist_min.gin | 

| random-dynamic | config |
|----------------|--------|
| on+PIBT        | config/competition/online_update/32x32_random_400_mixed_dist_traf_task.gin |
| off+PIBT       | config/competition/cnn_iter_update/32x32_random_400_mixed_dists.gin |
| on+GPIBT       | config/traffic_mapf/random_dist.gin |
| [p-on]+GPIBT   | config/traffic_mapf/period_online/random_dist.gin |
| off+GPIBT      | config/traffic_mapf/offline/random_dist.gin |

| warehouse-large-static | config |
|------------------------|--------|
| on+PIBT                | config/competition/online_update/60x100_wnarrow_1800.gin |
| off+PIBT               | config/competition/cnn_iter_update/60x100_warehouse_1800.gin |
| on+GPIBT               | config/traffic_mapf/warehouse_60x100.gin |
| [p-on]+GPIBT           | config/traffic_mapf/period_online/warehouse_60x100.gin |
| off+GPIBT              | config/traffic_mapf/offline/warehouse_60x100.gin |

| warehouse-large-dynamic | config |
|-------------------------|--------|
| on+PIBT                 | config/competition/online_update/60x100_w_1800_dist_mixed.gin |
| off+PIBT                | config/competition/cnn_iter_update/60x100_w_1800_dist.gin |
| on+GPIBT                | config/traffic_mapf/warehouse_60x100.gin |
| [p-on]+GPIBT            | config/traffic_mapf/warehouse_60x100_dist_mixed.gin |
| off+GPIBT               | config/traffic_mapf/offline/warehouse_60x100_dist_mixed.gin |

| game-static | config |
|-------------|--------|
| on+GPIBT    | config/traffic_mapf/ost.gin |
| off+GPIBT   | config/traffic_mapf/offline/ost.gin |

| game-dynamic | config |
|--------------|--------|
| on+GPIBT     | config/traffic_mapf/ost_dist.gin |
| off+GPIBT    | config/traffic_mapf/offline/ost_dist.gin |

| maze-static | config |
|-------------|--------|
| on+PIBT     | config/competition/online_update/32x32_maze_400_task_traf.gin |
| off+PIBT    | config/competition/cnn_iter_update/32x32_maze_400.gin |

| game-small-static | config |
|-------------------|--------|
| on+PIBT           | config/competition/online_update/97x97_ost_1600.gin |
| off+PIBT          | config/competition/cnn_iter_update/97x97_ost_1600.gin |

## Evaluation

### Preprocessing
Before evaluating the guidance policy, we need to do some preprocessing of logging data. 

1. **Guidance Policy Extraction**
under CMAES/ directory. Find the ``scripts/eval/extract_net_weighs.sh``. Filling in the ``PROJECT_DIR``, ``LOGDIR``, and ``DOMAIN``. As mentioned before ``PROJECT_DIR`` is the absolute path of the base directory. ``LOGDIR`` is the optimization logging directory saved in ``CMAES/logs/`` by default. ``DOMAIN`` can be selected from ``competition`` and ``trafficMAPF``, corresponding to PIBT an GPIBT settings, respectively. After setting all values, run the following command. 
   ```bash
   sh scripts/eval/extract_net_weights.sh
   ```
### Evaluation
(under CMAES/ directory) Find ``scripts/eval/eval_multiprocess.sh``. Filling the ``PROJECT_DIR`` and ``LOGDIR``. If using GPIBT ckpt, replace ``env_search/competition/multi_process_eval.py`` to ``env_search/traffic_mapf/multi_process_eval.py``. 

## Baseline Results
### Offline 
1. optimization: same as the guidance policy
2. evaluation: same as the guidance policy
3. Note: For **off+GPIBT**, you should manually extract the offline guidance graph using scripts ``scripts/eval/extract_trafficflow_off_weights.sh`` before evaluation.

### GPIBT
(under GPIBT/ directory)
create a virtual environment of python 3.9
1. compile the GPIBT using ``compile.sh``(Remember replace the ``BASE_DIR`` to the absolute path of the base diretory).
2. run ``multi_process_eval.py``. 

## Visualization
Visualization for congestion, weights ratio, and the number of reached goal per timestep. 

under CMAES/ directory. Find the ``scripts/eval/vis_pibt.sh`` and ``scripts/eval/vis_gpibt.sh``. 
Filling in related configuration option. 
