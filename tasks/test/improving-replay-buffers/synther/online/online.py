import warnings

warnings.filterwarnings("ignore")

import gin
import gym
from gym.wrappers.flatten_observation import FlattenObservation


def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)

    return env


def get_time_limit(env: gym.Env):
    if hasattr(env, 'spec'):
        if hasattr(env.spec, 'max_episode_steps'):
            return env.spec.max_episode_steps
    if hasattr(env, 'env'):
        return get_time_limit(env.env)
    if hasattr(env, 'unwrapped'):
        return get_time_limit(env.unwrapped)
    else:
        raise ValueError("Cannot find time limit for env")


@gin.configurable
def run(env_name: str):
    env = wrap_gym(gym.make(env_name))
    _ = get_time_limit(env)
    print("Skeleton RL entrypoint loaded.")
    print("Implement or plug in your baseline training loop here.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--gin_config_files', nargs='*', type=str, default=['config/online/sac.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[])
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)
    run(args.env)


