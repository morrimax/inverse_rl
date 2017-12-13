from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from inverse_rl.envs.env_utils import CustomGymEnv

from inverse_rl.utils.log_utils import rllab_logdir

def main():
    env = TfEnv(CustomGymEnv('PointMazeLeft-v0'))
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = TRPO(
        env=env,
        policy=policy,
        n_itr=2000,
        batch_size=10000,
        max_path_length=100,
        discount=0.99,
        store_paths=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

    with rllab_logdir(algo=algo, dirname='data/point'):
        algo.train()

if __name__ == "__main__":
    main()
