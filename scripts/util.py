import numpy as np
from inverse_rl.envs.env_utils import CustomGymEnv
from sandbox.rocky.tf.envs.base import TfEnv


def test_pointmaze(policy):
    test_env = TfEnv(CustomGymEnv('PointMazeRight-v0'))
    for i in range(5):
        done = False
        s = test_env.reset()
        reward = 0
        steps = 0
        while not done:
            a = np.random.choice(policy.shape[1], policy[s])
            s_, r, _, done = test_env.step(a)
            steps += 1
            reward += r
        print('Average episode reward is {}'.format(reward / steps))
