import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def main():
	airl = get_returns('data/ant_airl/progress.csv')
	gcl = get_returns('data/ant_gcl/progress.csv')
	gail = get_returns('data/ant_gail/progress.csv')

	sns.set_style('dark')

	plt.plot(airl, label='airl')
	plt.plot(gcl, label='gcl', linestyle='--')
	plt.plot(gail, label='gail', linestyle=':')

	plt.xlabel('Episode')
	plt.ylabel('Average Return')
	plt.title('IRL Performance')
	plt.legend()

	plt.savefig('ant_irl_performance')

def get_returns(filename):
	df = pd.read_csv(filename)
	return df['OriginalTaskAverageReturn'].as_matrix()

if __name__ == '__main__':
	main()