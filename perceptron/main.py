import numpy as np

def	initialize():
	inputs = np.array([
		[0, 0, 1],
		[1, 1, 1],
		[1, 0, 1],
		[0, 1, 1]
	])

	outputs = np.array([
		[0, 1, 1, 0]
	]).T
	
	np.random.seed(42)
	synaptic_weights = 2 * np.random.random((3, 1)) - 1

	print("Random starting synaptic weights: ")
	print(synaptic_weights)
	return (inputs, outputs, synaptic_weights)

def	sigmoid(x):
	return (1 / (1 + np.exp(-x)))

def derivative_sigmoid(x):
	return (x * (1 - x))

def train(inputs, outputs, synaptic_weights, iterations=10000):
	trained_synaptic_weights = synaptic_weights
	for i in range(iterations):
		trained_outputs = sigmoid(np.dot(inputs, synaptic_weights))
		error = outputs - trained_outputs
		adjustments = error * derivative_sigmoid(trained_outputs)
		trained_synaptic_weights += np.dot(inputs.T, adjustments)
	return (trained_synaptic_weights, trained_outputs)

def	print_result(trained_synaptic_weights, trained_outputs):
	print()
	print("Synaptic weights after training: ")
	print(trained_synaptic_weights)
	print()
	print("Outputs after training: ")
	print(trained_outputs)

def main():
	inputs, outputs, synaptic_weights = initialize()	
	trained_synaptic_weights, trained_outputs = train(inputs, outputs, synaptic_weights, 10000)
	print_result(trained_synaptic_weights, trained_outputs)

if __name__ == "__main__":
	main()