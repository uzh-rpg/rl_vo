#!/usr/bin/env python3
import os
import numpy as np
import torch
import hydra
from datetime import datetime
from stable_baselines3.common.utils import get_linear_fn

from env.svo_wrapper import VecSVOEnv
from rl_algorithms.ppo import PPO
from policies.attention_policy import CustomActorCriticPolicy


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(config):
    torch.init_num_threads()

    if config.vo_algorithm == "SVO":
        env = VecSVOEnv(
            config.svo_params_file, config.svo_calib_file,
            config.dataset_dir, config.n_envs, reward_config=config.agent.reward,
            mode='train', initialize_glog=True)
        val_env = VecSVOEnv(
            config.svo_params_file, config.svo_calib_file, config.dataset_dir, 32,
            reward_config=config.agent.reward, mode='val', initialize_glog=False)
    else:
        assert False, "Unknown VO algorithm"

    policy = CustomActorCriticPolicy
    encoder_kwargs = dict(
        variable_feature_dim=3,
        obs_dim_variable=env.agent_obs_dim_variable,
        obs_dim_fixed=env.agent_obs_dim_fixed,
        critique_dim=env.critique_dim,
    )
    policy_kwargs = dict(
        encoder_kwargs=encoder_kwargs,
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=-0.0,
    )

    # set random seed
    configure_random_seed(config.seed, env=env)

    # save the configuration and other files
    if not config.wandb_logging:
        log_dir = os.path.join(config.log_path, datetime.now().strftime('%b%d_%H-%M-%S'))
    elif config.wandb_group is not None:
        log_dir = os.path.join(config.log_path, config.wandb_group, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + config.wandb_tag)
    else:
        log_dir = os.path.join(config.log_path, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + config.wandb_tag)
    if config.wandb_logging:
        os.makedirs(os.path.join(log_dir, 'wandb'))

    # check if gou is available, if not use cpu
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    if config.policy_path is not None:
        env.load_rms(config.policy_path[:-4] + '_rms.npz')
        val_env.load_rms(config.policy_path[:-4] + '_rms.npz')

    model = PPO(
        tensorboard_log=None,
        log_dir=log_dir,
        policy=policy,
        policy_kwargs=policy_kwargs,
        env=env,
        n_epochs=config.agent.n_epochs,
        gae_lambda=config.agent.gae_lambda,
        gamma=config.agent.gamma,
        n_steps=config.agent.n_steps,
        ent_coef=config.agent.ent_coef,
        vf_coef=config.agent.vf_coef,
        max_grad_norm=config.agent.max_grad_norm,
        batch_size=config.agent.batch_size,
        learning_rate=get_linear_fn(3e-4, 3e-5, 1.0),
        clip_range=0.2,
        use_sde=config.agent.use_sde,
        verbose=1,
        seed=config.seed,
        wandb_logging=config.wandb_logging,
        wandb_tag=config.wandb_tag,
        wandb_group=config.wandb_group,
        config=config,
        device=device,
    )

    # Load policy
    if config.policy_path is not None:
        state_dict = torch.load(config.policy_path, map_location=device, weights_only=False)["state_dict"]
        model.policy.load_state_dict(state_dict, strict=False)
        model.policy.to(device)

    model.learn(
        total_timesteps=int(config.total_timesteps),
        log_interval=100,
        eval_interval=config.val_interval,
        val_env=val_env,
    )


if __name__ == "__main__":
    main()
