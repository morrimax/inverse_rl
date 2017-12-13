import ant_airl as airl
import ant_data_collect as collect
import ant_gail as gail
import ant_gcl as gcl
import ant_traj as traj
import numpy as np
import pandas as pd

def mean_stat(progress_file, statistic, lookback=20):
    df = pd.read_csv(progress_file)
    return np.mean(df.tail(lookback)[statistic])

def main(num_runs=1):
    algos = ['airl', 'gail', 'gcl']
    scores = dict(zip(algos, [[]] * len(algos)))

    for i in range(num_runs):
        airl.main()
        gail.main()
        gcl.main()
        # traj.main()

        for algo in algos:
            stat = 'OriginalTaskAverageReturn'
            scores[algo].append(mean_stat('data/ant_{}/progress.csv'.format(algo), statistic=stat))

    print (scores)
    for algo, score in scores.items():
        print('Algorithm ({}) mean performance: {}'.format(algo, np.mean(score)))

if __name__ == "__main__":
    main()