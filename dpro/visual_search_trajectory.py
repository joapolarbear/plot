import numpy as np
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_data(path):
	with open(path, 'r') as fp:
		lines = fp.read().split('\n')

	data = []
	start_t = None
	for line in lines:
		try:
			t, per = line.split(":")
		except:
			continue
		t = float(t)
		per_list = [float(x[:6]) for x in per.split(",")]
		print(per_list)
		if start_t is None:
			start_t = t
		data.append([(t-start_t)/3600.] + per_list)

	# shape = (n_dim, n_samples), dim: time, step, cur_speedup, best_speedup
	# data = sorted(data, key = lambda x: x[1])
	data = np.array(data).T
	return data


fig = plt.figure()

if os.path.isdir(sys.argv[1]):
	ax = fig.add_subplot(111)
	# names = sys.argv[2].split(",")
	_, _, files = list(os.walk(sys.argv[1]))[0]
	for file in sorted(files):
		try:
			data = read_data(os.path.join(sys.argv[1], file))
		except:
			print(file)
			raise
		ax.plot(data[1], data[2], label=file.split(".txt")[0])
	plt.legend()
	plt.xlabel('Step')
	plt.ylabel('Cur Performance Speedup (%)')
	plt.show()

	
else:
	data = read_data(sys.argv[1])
	print(data)

	
	ax = fig.add_subplot(221)
	ax.plot(data[0], data[2])
	plt.xlabel('Time (h)')
	plt.ylabel('Cur Performance Speedup (%)')

	ax = fig.add_subplot(222)
	ax.plot(data[1], data[2])
	plt.xlabel('Step')
	plt.ylabel('Cur Performance Speedup (%)')

	ax = fig.add_subplot(223)
	ax.plot(data[0], data[3])
	plt.xlabel('Time (h)')
	plt.ylabel('Best Performance Speedup (%)')

	ax = fig.add_subplot(224)
	ax.plot(data[1], data[3])
	plt.xlabel('Step')
	plt.ylabel('Best Performance Speedup (%)')

	# ax.set_xlabel('Tensor ID')
	# ax.set_ylabel('# of Steps')
	# ax.set_zlabel('Group ID')
	plt.show()