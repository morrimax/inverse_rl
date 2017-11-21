import argparse
import numpy as np
import pandas as pd

def main(progress_file, statistic, lookback):
    df = pd.read_csv(progress_file)
    print('Mean performance is {}'.format(
        np.mean(df.tail(lookback)[statistic])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--progress_file', required=True)
    parser.add_argument('--statistic', default='OriginalTaskAverageReturn')
    parser.add_argument('--lookback', type=int, default=5)
    args = parser.parse_args()
    main(args.progress_file, args.statistic, args.lookback)