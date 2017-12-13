import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from util import test_pointmaze

from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.imitation_learning import GCLDiscrim
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts
from inverse_rl.models.architectures import disentangled_net

def main():
    env = TfEnv(CustomGymEnv('PointMazeLeft-v0'))
    
    experts = load_latest_experts('data/point', n=50)

    irl_model = GCLDiscrim(
        env_spec=env.spec,
        expert_trajs=experts,
        discrim_arch=disentangled_net)
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=2000,
        batch_size=10000,
        max_path_length=100,
        discount=0.99,
        store_paths=True,
        discrim_train_itrs=50,
        irl_model_wt=1.0,
        entropy_weight=0.1,
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

    with rllab_logdir(algo=algo, dirname='data/point_airl'):
        with tf.Session():
            algo.train()
            test_pointmaze(sess.run(policy))

if __name__ == "__main__":
    main()
