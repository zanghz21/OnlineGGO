from operator import index
import os
from typing import List
import gin
import json
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import fire
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from logdir import LogDir
from pathlib import Path
from pprint import pprint
from env_search.analysis.utils import (load_experiment, load_metrics,
                                       load_archive_gen)
from env_search.analysis.visualize_env import (visualize_kiva,
                                               visualize_competition)
from env_search.utils.logging import setup_logging
from env_search.mpl_styles.utils import mpl_style_file
from env_search.utils import (
    set_spines_visible,
    kiva_env_number2str,
    kiva_env_str2number,
    read_in_kiva_map,
    write_map_str_to_json,
    competition_env_str2number,
    write_iter_update_model_to_json,
    min_max_normalize,
    get_n_valid_vertices,
)

mpl.use("agg")

# set matplotlib params
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIG_HEIGHT = 10


def get_tile_usage_vmax(env_h, env_w):
    if env_h * env_w == 9 * 16 or env_h * env_w == 9 * 20:
        vmax = 0.03
    elif env_h * env_w == 17 * 16 or env_h * env_w == 17 * 20:
        vmax = 0.02
    elif env_h * env_w == 33 * 36:
        vmax = 0.01
    elif env_h * env_w == 101 * 102:
        vmax = 0.002
    else:
        vmax = 0.04
    return vmax


def plot_tile_usage(
    tile_usage,
    env_h,
    env_w,
    fig,
    ax_tile_use,
    ax_tile_use_cbar,
    logdir,
    filenames: List = ["tile_usage.pdf", "tile_usage.svg", "tile_usage.png"],
    dpi=300,
):
    # Plot tile usage
    tile_usage = tile_usage.reshape(env_h, env_w)

    sns.heatmap(
        tile_usage,
        square=True,
        cmap="Reds",
        ax=ax_tile_use,
        cbar_ax=ax_tile_use_cbar,
        cbar=ax_tile_use_cbar is not None,
        rasterized=False,
        annot_kws={"size": 30},
        linewidths=1,
        linecolor="black",
        xticklabels=False,
        yticklabels=False,
        vmin=0,
        vmax=get_tile_usage_vmax(env_h, env_w),
        cbar_kws={
            "orientation": "horizontal",
            "shrink": 0.7,
        } if ax_tile_use_cbar is not None else None,
    )

    if ax_tile_use_cbar is not None:
        cbar = ax_tile_use.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)

    set_spines_visible(ax_tile_use)
    ax_tile_use.figure.tight_layout()
    fig.tight_layout()

    for filename in filenames:
        fig.savefig(logdir.file(filename), dpi=dpi)


def get_figsize_qd(w_mode=True, domain="kiva"):
    # Decide figsize based on size of map
    if domain == "kiva":
        env_h = gin.query_parameter("WarehouseManager.lvl_height")
        env_w = gin.query_parameter("WarehouseManager.lvl_width")
    elif domain == "competition":
        env_h = gin.query_parameter("CompetitionManager.lvl_height")
        env_w = gin.query_parameter("CompetitionManager.lvl_width")

    if env_h * env_w == 9 * 12 or env_h * env_w == 9 * 16:
        if w_mode:
            figsize = (8, 8)
        else:
            figsize = (8, 12)
    elif env_h * env_w == 17 * 12 or env_h * env_w == 17 * 16:
        figsize = (8, 16)
    elif env_h * env_w == 33 * 32 or env_h * env_w == 33 * 36:
        figsize = (8, 16)
    else:
        figsize = (8, 8)

    return figsize


def get_figsize_sim(env_np):
    # Decide figsize based on size of map
    env_h, env_w = env_np.shape
    if env_h * env_w == 9 * 16 or env_h * env_w == 9 * 20:
        figsize = (8, 8)
    elif env_h * env_w == 17 * 16 or env_h * env_w == 17 * 20:
        figsize = (8, 16)
    elif env_h * env_w == 33 * 36:
        figsize = (8, 16)
    else:
        figsize = (8, 16)

    return figsize


def tile_usage_heatmap_from_single_run(logdir: str, dpi=300, domain="kiva"):
    """
    Plot tile usage with map layout from a single run of warehouse simulation.
    """
    # Read in map
    map_filepath = os.path.join(logdir, "map.json")
    if domain == "kiva":
        map, map_name = read_in_kiva_map(map_filepath)
        env_np = kiva_env_str2number(map)

    # Create plot
    # grid_kws = {"height_ratios": (0.475, 0.475, 0.05)}
    # fig, (ax_map, ax_tile_use, ax_tile_use_cbar) = plt.subplots(
    #     3,
    #     1,
    #     figsize=get_figsize_sim(env_np),
    #     gridspec_kw=grid_kws,
    # )
    n_row, n_col = env_np.shape
    fig, ax_tile_use = plt.subplots(figsize=(FIG_HEIGHT * n_col / n_row,
                                             FIG_HEIGHT))

    # # Plot map
    # if domain == "kiva":
    #     visualize_kiva(env_np, ax=ax_map, dpi=300)

    # Read in result and plot tile usage
    results_dir = os.path.join(logdir, "results")
    for sim_dir in tqdm(os.listdir(results_dir)):
        sim_dir_comp = os.path.join(results_dir, sim_dir)
        result_filepath = os.path.join(sim_dir_comp, "result.json")
        with open(result_filepath, "r") as f:
            result_json = json.load(f)
        tile_usage = result_json["tile_usage"]
        plot_tile_usage(
            np.array(tile_usage),
            env_np.shape[0],
            env_np.shape[1],
            fig,
            ax_tile_use,
            None,
            LogDir(map_name, custom_dir=sim_dir_comp),
            filenames=[
                f"tile_usage_{map_name}.pdf",
                f"tile_usage_{map_name}.svg",
                f"tile_usage_{map_name}.png",
            ],
            dpi=dpi,
        )


def tile_usage_heatmap_from_qd(
    logdir: str,
    gen: int = None,
    index_0: int = None,
    index_1: int = None,
    dpi: int = 300,
    mode: str = None,
    domain: str = "kiva",
):
    """
    Plot tile usage with map layout from a QD experiment.
    """
    logdir = load_experiment(logdir)
    gen = load_metrics(logdir).total_itrs if gen is None else gen
    archive = load_archive_gen(logdir, gen)
    try:
        df = pd.read_pickle(logdir.file(f"archive/archive_{gen}.pkl"))
    except FileNotFoundError:
        df = pd.read_pickle(logdir.file(f"archive/archive_.pkl"))
    global_opt_weights = None

    # Convert index to index_0 and index_1 --> pyribs 0.4.0/0.5.0 compatibility
    # issue
    if "index_0" not in df and "index" in df:
        all_grid_index = archive.int_to_grid_index(df["index"])
        df["index_0"] = all_grid_index[:, 0]
        df["index_1"] = all_grid_index[:, 1]

    if index_0 is not None and index_1 is not None:
        to_plots = df[(df["index_0"] == index_0) & (df["index_1"] == index_1)]
        if to_plots.empty:
            raise ValueError("Specified index has no solution")
    elif mode == "extreme":
        # Plot the "extreme" points in the archive
        # Choose the largest measure1 or measure2. If tie, choose the one with
        # larger objective.
        # df.loc[df["index_0"]==df["index_0"].max()]["objective"].idxmax()

        index_0_max = df.loc[df["index_0"] ==
                             df["index_0"].max()]["objective"].idxmax()
        index_0_min = df.loc[df["index_0"] ==
                             df["index_0"].min()]["objective"].idxmax()
        index_1_max = df.loc[df["index_1"] ==
                             df["index_1"].max()]["objective"].idxmax()
        index_1_min = df.loc[df["index_1"] ==
                             df["index_1"].min()]["objective"].idxmax()

        # Add global optimal env
        global_opt = df["objective"].idxmax()
        optimize_wait = True  # Default
        iterative_update = False  # Default
        # if domain in ["kiva"]:
        if domain == "kiva":
            # Read in base map
            base_map_path = gin.query_parameter(
                "WarehouseManager.base_map_path")
            optimize_wait = gin.query_parameter(f"%optimize_wait")
            
            try:
                iterative_update = gin.query_parameter(
                    "WarehouseManager.iterative_update")
            except ValueError:
                iterative_update = False

            with open(base_map_path, "r") as f:
                base_map_json = json.load(f)

            opt_edge_weights = df.iloc[global_opt]["metadata"][
                "warehouse_metadata"]["edge_weights"]
            opt_wait_costs = df.iloc[global_opt]["metadata"][
                "warehouse_metadata"]["wait_costs"]

        elif domain == "competition":
            base_map_path = gin.query_parameter("%base_map_path")
            with open(base_map_path, "r") as f:
                base_map_json = json.load(f)
            try:
                iterative_update = gin.query_parameter(
                    "CompetitionManager.iterative_update")
            except ValueError:
                iterative_update = False

            opt_edge_weights = df.iloc[global_opt]["metadata"][
                "competition_metadata"]["edge_weights"]
            opt_wait_costs = df.iloc[global_opt]["metadata"][
                "competition_metadata"]["wait_costs"]
            lb, ub = gin.query_parameter("%bounds")
            opt_edge_weights = min_max_normalize(opt_edge_weights, lb, ub)
            opt_wait_costs = min_max_normalize(opt_wait_costs, lb, ub)
        elif domain == "trafficMAPF":
            pass
        else:
            raise ValueError(f"Unknown domain: {domain}")

        # Write update model
        if iterative_update or domain == "trafficMAPF":
            # Write optimal trained update model param
            opt_update_model_param = df.filter(
                regex=("solution.*")).iloc[global_opt].to_list()

            if domain == "trafficMAPF":
                update_model_type = gin.query_parameter("TrafficMAPFConfig.net_type")
            elif domain == "competition":
                update_model_type = gin.query_parameter(
                "CompetitionConfig.iter_update_model_type")
            elif domain == "kiva":
                update_model_type = "cnn"

            write_iter_update_model_to_json(
                logdir.pfile("optimal_update_model.json"),
                opt_update_model_param,
                update_model_type,
            )

        if domain == "trafficMAPF":
            return
        # Write optimal weights
        global_opt_weights = [*opt_wait_costs, *opt_edge_weights]
        global_opt_weights = [float(x) for x in global_opt_weights]

        # Save global_optimal solution
        write_map_str_to_json(
            logdir.pfile("optimal_weights.json"),
            base_map_json["layout"],
            "global_optimal",
            domain,
            weight=True,
            weights=global_opt_weights,
            optimize_wait=optimize_wait,
        )

        to_plots = df.iloc[[
            index_0_max,
            index_0_min,
            index_1_max,
            index_1_min,
            global_opt,
        ]]

    with mpl_style_file("tile_usage_heatmap.mplstyle") as f:
        with plt.style.context(f):
            plot_idx = 0
            for _, to_plot in to_plots.iterrows():
                index_0 = to_plot["index_0"]
                index_1 = to_plot["index_1"]
                obj = to_plot["objective"]
                if domain == "kiva":
                    metadata = to_plot["metadata"]["warehouse_metadata"]
                elif domain == "competition":
                    metadata = to_plot["metadata"]["competition_metadata"]
                throughput = np.mean(metadata["throughput"])

                # The last one is the global optimal
                if plot_idx == len(to_plots) - 1:
                    print("Optimal in archive: ", end="")
                plot_idx += 1

                print(
                    f"Index ({index_0}, {index_1}): objective = {obj}, throughput = {throughput}"
                )
                if domain == "kiva":
                    w_mode = gin.query_parameter("WarehouseManager.w_mode")
                elif domain in ["competition"]:
                    w_mode = False

                grid_kws = {"height_ratios": (0.475, 0.475, 0.05)}
                fig, (ax_map, ax_tile_use, ax_tile_use_cbar) = plt.subplots(
                    3,
                    1,
                    figsize=get_figsize_qd(w_mode, domain),
                    gridspec_kw=grid_kws,
                )
                

def main(
    logdir: str,
    logdir_type: str = "qd",  # "qd" or "sim"
    gen: int = None,
    index_0: int = None,
    index_1: int = None,
    dpi: int = 300,
    mode: str = None,
    domain: str = "kiva",
):
    if logdir_type == "qd":
        tile_usage_heatmap_from_qd(
            logdir=logdir,
            gen=gen,
            index_0=index_0,
            index_1=index_1,
            dpi=dpi,
            mode=mode,
            domain=domain,
        )
    elif logdir_type == "sim":
        tile_usage_heatmap_from_single_run(logdir, dpi=dpi, domain=domain)


if __name__ == "__main__":
    fire.Fire(main)
