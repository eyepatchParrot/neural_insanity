# TODO get this working with a single boolean function
# TODO auto generate the number of connections and layers needed
# TODO get this working with an arbitrary boolean expression


# a neuron has a bias and edges to other neurons. Since a neuron depends only
# on neurons in previous layers, calculate one layer at a time.

# binary conversion has to be based on weight in order for this to be prunable

# the reason this works is because overfitting is OK in the case where we're
# guaranteed to have enough neurons to fully represent the function

import random
import itertools
import math

alpha = 0.1

def dot_product(a, b):
	sum = 0.0
	for i in xrange(len(a)):
		sum = sum + a[i] * b[i]
	return sum

def sigmoid(x):
	return 1.0 / (1.0 + math.exp(-x))

class Neuron():
	def __init__(self, numWeights):
		self.weights = []

		for i in xrange(numWeights + 1):
			self.weights.append(random.uniform(-1, 1))

	def __str__(self):
		return "{\n\t" + str(self.weights) + "\n}"

	def activate(self, inputs):
		t_inputs = list(inputs)
		t_inputs.append(1)
		return sigmoid(dot_product(t_inputs, self.weights))

	# d_w_(j,k) = alpha * a_j * d_k
	# d_k = (T - O) * O * (1 - O)
	def trainjk(self, t, o, inputs):
		t_inputs = list(inputs)
		t_inputs.append(1)
		d_k = (t - o) * o * (1 - o)
		for i in xrange(len(self.weights)):
			d_w_k = alpha * t_inputs[i] * d_k
			self.weights[i] = self.weights[i] + d_w_k

	# d_w_(i,j) = alpha * a_i * d_j
	# d_j = a_j * (1 - a_j) * sum{w_(j,k) * d_k
#	def trainij(self, j, t, o, inputs, output):
#		d_j = output.weights[j] * output.d_k * (1-


def is_token(c):
	return c in 'abcdefghijklmnopqrstuvwxyz'

def eval(expr):
	stack = []
	for x in expr:
		if isinstance(x, bool):
			stack.append(x)
		else:
			if (x == '~'):
				a = stack.pop()
				stack.append(not a)
			else:
				b = stack.pop()
				a = stack.pop()
				if (x == '&&'):
					stack.append(a and b)
				elif (x == '||'):
					stack.append(a or b)
	return stack.pop()

def gen_training(s):
	s = s.split()
	print s
	# find unique tokens, and look at all possible combinations
	# then substitute in the values, evaluate, and append to the training set
	unique = filter(is_token, set(s))
	unique.sort()
	training = []

	for r in xrange(len(unique) + 1):
		for comb in itertools.combinations(unique, r):
			# substitute
			sensors = map(lambda x: x in comb, unique)
			expr = map(lambda x: x in comb if is_token(x) else x, s)
			t = eval(expr)
			training.append((t, sensors))
	return training
		
training = gen_training(raw_input())
#training = gen_training("a b &&")
_, sensors = training[0]
output = Neuron(len(sensors))

print(training)
print(output)

epoch = 0
accuracy = 0.0

while (accuracy < 0.9):
	# epoch
	correct = 0
	for t, inputs in training:
		o = output.activate(inputs)
		output.trainjk(t, o, inputs)
#		print("t " + str(t) + " o " + str(o))
#		print(output)
		if (abs(t - o) < 0.5):
			correct = correct + 1
	accuracy = float(correct) / len(training)
	print("epoch " + str(epoch) + " accuracy " + str(accuracy))
	epoch = epoch + 1

print(output)
