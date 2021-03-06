import pendulum_airl as airl
import pendulum_data_collect as collect
import pendulum_gail as gail
import pendulum_gcl as gcl
import pendulum_traj as traj
import numpy as np
import pandas as pd

def mean_stat(progress_file, statistic, lookback=20):
    df = pd.read_csv(progress_file)
    return np.mean(df.tail(lookback)[statistic])

def main(num_runs=1):
    algos = ['airl', 'gail', 'gcl', 'traj']
    scores = dict(zip(algos, [[]] * len(algos)))

    for i in range(num_runs):
        airl.main()
        gail.main()
        gcl.main()
        traj.main()

        for algo in algos:
            stat = 'OriginalTaskAverageReturn'
            scores[algo].append(mean_stat('data/pendulum_{}/progress.csv'.format(algo), statistic=stat))

    for algo, score in scores.items():
        print('Algorithm ({}) mean performance: {}'.format(algo, np.mean(score)))

if __name__ == "__main__":
    main()