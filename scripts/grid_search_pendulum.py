import pendulum_airl as airl
import pendulum_data_collect as collect
import pendulum_gail as gail
import pendulum_gcl as gcl
import pendulum_traj as traj
import json
import numpy as np
import pandas as pd

def mean_stat(
    progress_file, statistic='OriginalTaskAverageReturn', lookback=20):
    df = pd.read_csv(progress_file)
    return np.mean(df[statistic][-lookback:])

def run_irl(algo, example, discount):
    algo[1].main(example, discount)
    mean_performance = mean_stat(
        'data/pendulum_{}/progress.csv'.format(algo[0]))
    print('{} achieved performance of {} with parameters ({},{})'.format(
        algo[0], mean_performance, example, discount))
    return mean_performance

def main(num_runs=1):
    algos = [('airl', airl), ('gail', gail), ('gcl', gcl), ('traj', traj)]
    examples = [10, 25, 50, 100, 150]
    discounts = [0.8, 0.9, 0.95, 0.99, 0.995]
    # algos = [('airl', airl)]
    # examples = [10]
    # discounts = [0.8]

    scores = {}
    for algo in algos:
        for example in examples:
            for discount in discounts:
                if algo[0] not in scores:
                    scores[algo[0]] = {}
                if example not in scores[algo[0]]:
                    scores[algo[0]][example] = {}
                scores[algo[0]][example][discount] = run_irl(
                    algo, example, discount)

    with open('grid_search.json', 'w') as f:
        f.write(json.dumps(scores))

if __name__ == "__main__":
    main()