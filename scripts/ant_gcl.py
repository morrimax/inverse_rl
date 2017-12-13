import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv


from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.imitation_learning import GCLDiscrim
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts

def main():
    env = TfEnv(GymEnv('Ant-v1', record_video=False, record_log=False))
    
    experts = load_latest_experts('data/ant', n=50)

    irl_model = GCLDiscrim(env_spec=env.spec, expert_trajs=experts)
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=2000,
        batch_size=10000,
        max_path_length=1000,
        discount=0.995,
        store_paths=True,
        discrim_train_itrs=50,
        irl_model_wt=1.0,
        entropy_weight=0.1, # this should be 1.0 but 0.1 seems to work better
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

    with rllab_logdir(algo=algo, dirname='data/ant_gcl'):
        with tf.Session():
            algo.train()

if __name__ == "__main__":
    main()
