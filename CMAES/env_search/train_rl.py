import argparse
import gin
from env_search.iterative_update.envs.env import CompetitionIterUpdateEnv
from env_search.iterative_update.envs.online_env import CompetitionOnlineEnv
from env_search.competition.config import CompetitionConfig
import wandb
from env_search.competition.update_model.utils import Map
from env_search.utils import get_n_valid_edges, get_n_valid_vertices
from env_search.utils.logging import get_current_time_str
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_search.rl_modules.config import RLConfig
from env_search.rl_modules.features_extractors import CNNFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback
import os
from functools import partial


def parseargs(parser: argparse.ArgumentParser):
    parser.add_argument('--piu_config_file', type=str)
    parser.add_argument('--n_workers', type=int)
    args = parser.parse_args()
    gin.parse_config_file(args.piu_config_file, skip_unknown=True)
    return args

def preprocessing(map_path):
    comp_map = Map(map_path)
    n_e = get_n_valid_edges(comp_map.graph, True, "competition")
    n_v = get_n_valid_vertices(comp_map.graph, "competition")
    return n_e, n_v

def create_model(env, rl_cfg:RLConfig):
    policy_kwargs = {"features_extractor_class": CNNFeaturesExtractor}
    model = PPO("CnnPolicy",
                env,
                verbose=1,
                policy_kwargs=policy_kwargs, 
                learning_rate=rl_cfg.lr, 
                n_steps=rl_cfg.n_steps,
                batch_size=16,
                n_epochs=rl_cfg.ppo_epoch, 
                gamma=rl_cfg.gamma,
                clip_range=rl_cfg.clip_range,
                max_grad_norm=10., 
                ent_coef=rl_cfg.entropy_coef, 
                seed=rl_cfg.seed, 
                tensorboard_log=rl_cfg.tb_dir, 
    )
    return model


def train(args):
    time_str = get_current_time_str()
    env_cfg = CompetitionConfig()
    rl_cfg = RLConfig()
    rl_cfg.tb_dir = os.path.join(rl_cfg.tb_base_dir, time_str)
    
    n_e, n_v = preprocessing(env_cfg.map_path)
    
    def make_OnlineEnv(seed):
        return CompetitionOnlineEnv(n_v, n_e, env_cfg, seed)
    env_list = [partial(make_OnlineEnv, seed=i+1) for i in range(args.n_workers)]
    env = SubprocVecEnv(env_list)

    model = create_model(env, rl_cfg)
    
    callback_list = []
    
    if rl_cfg.eval_freq > 0:
        eval_env = CompetitionOnlineEnv(n_v, n_e, env_cfg, seed=0)
        eval_callback = EvalCallback(eval_env, 
                best_model_save_path=os.path.join(rl_cfg.tb_dir, 'ckpts'), 
                n_eval_episodes=rl_cfg.eval_episodes, 
                log_path=os.path.join(rl_cfg.tb_dir, 'eval_logs'), 
                eval_freq=rl_cfg.eval_freq,
                deterministic=True, render=False)
        callback_list.append(eval_callback)
    
    all_callback = CallbackList(callback_list)
    
    run = wandb.init(
        name=time_str, 
        project="OnlineGGO-RL", 
        # mode="disabled", 
        config={"env": env_cfg, "rl": rl_cfg}, 
        sync_tensorboard=True
    )
    model.learn(total_timesteps=rl_cfg.total_timesteps, log_interval=rl_cfg.log_interval, callback=all_callback)
    env.close()
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parseargs(parser)
    train(args)
