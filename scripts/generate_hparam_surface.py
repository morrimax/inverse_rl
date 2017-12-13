import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

def load(algorithm, input_file):
	with open(input_file) as f:
		example_dict = json.load(f)[algorithm.lower()]
	examples = np.zeros((5, 5))
	discounts = np.zeros((5, 5))
	scores = np.zeros((5, 5))
	for i, (example, discount_dict) in enumerate(example_dict.items()):
		for j, (discount, score) in enumerate(discount_dict.items()):
			examples[i][j] = example
			discounts[i][j] = discount
			scores[i][j] = score
	return examples, discounts, scores

def plot(examples, discounts, scores, algorithm, output_file):
	sb.set_style('dark')
	fig = plt.figure()
	ax = Axes3D(fig)
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(examples, discounts, scores)
	ax.set_xlabel('Number of Examples')
	ax.set_ylabel('Discount Factor')
	ax.set_zlabel('Average Return')
	if algorithm == 'GCL':
		algorithm = 'AIRL (non-robust)'
	elif algorithm == 'TRAJ':
		algorithm = 'GAN-GCL'
	ax.set_title('{} Hyperparameter Surface'.format(algorithm))
	plt.show()

def main(algorithm, input_file, output_file):
	examples, discounts, scores = load(algorithm, input_file)
	plot(examples, discounts, scores, algorithm, output_file)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--algorithm', required=True)
	parser.add_argument('--input_file', required=True)
	parser.add_argument('--output_file', required=True)
	args = parser.parse_args()
	main(args.algorithm, args.input_file, args.output_file)
