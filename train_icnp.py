import argparse

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO

from algos.equivariant_policy import EquivariantActorCriticPolicy, IdentityFeatureExtractor
from envs.icnp_sa import ICNP2021Env

if __name__ == "__main__":
    # 创建一个ArgumentParser对象，用于解析命令行参数。
    parser = argparse.ArgumentParser(description="Process PPO training arguments.")
    # 这些行为命令行解析器添加了参数选项。这些参数允许用户从命令行定制化训练配置。如果用户没有在命令行上指定这些参数的值，代码将使用默认值。
    parser.add_argument("--env_type", type=str, default="nsfnet")
    parser.add_argument("--episode_length", type=int, default=10)
    parser.add_argument("--traffic_profile", type=str, default="uniform")
    parser.add_argument("--n_cpu", type=int, default=8)
    parser.add_argument("--log_dir", type=str, default="./results/icnp/")
    # 解析命令行输入的参数。
    args = parser.parse_args()

    # 从解析的命令行参数中获取日志目录的路径，并将其赋值给monitor_dir变量。
    monitor_dir = args.log_dir

    # 定义一个字典，其中包含配置程序性能监视器的关键字参数。
    monitor_kwargs = {"info_keywords": ("starting_max_load", "max_load", "n_accepted")}
    # 创建包含环境配置的字典，该配置将基于命令行参数。
    env_kwargs = {"env_type": args.env_type, "episode_length": args.episode_length, "traffic_profile": args.traffic_profile}
    # 调用make_vec_env函数来创建一个向量化环境，它是用于并行运行多个环境实例的容器。这里使用SubprocVecEnv类，这个类是用来在子进程中创建多个环境的，并且配置了监视器，它将跟踪特定的性能指标。
    env = make_vec_env(
        ICNP2021Env,
        n_envs=args.n_cpu,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=monitor_dir,
        monitor_kwargs=monitor_kwargs,
        env_kwargs=env_kwargs,
        seed=0,
    )
    # 定义策略参数中需要的关键字参数的字典。IdentityFeatureExtractor是用于提取特征的类。
    policy_kwargs = dict(features_extractor_class=IdentityFeatureExtractor, features_extractor_kwargs=dict(features_dim=2))
    # 实例化一个PPO代理/智能体，使用EquivariantActorCriticPolicy策略，传入环境，指定学习率、步长和日志级别等参数。
    agent = PPO(EquivariantActorCriticPolicy, env, learning_rate=0.0003, n_steps=128, verbose=1, policy_kwargs=policy_kwargs)
    # 开始训练过程，持续300000个时间步长。
    agent.learn(300000)
    # 训练完成后，保存训练好的模型到指定的监控目录。
    agent.save(monitor_dir + "model")
