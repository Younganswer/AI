import numpy as np

def	sigmoid(x):
	return (1 / (1 + np.exp(-x)))

def train(inputs, outputs, synaptic_weights, iterations=1):
	for i in range(iterations):
		input_layer = inputs
		outputs = sigmoid(np.dot(input_layer, synaptic_weights))
	return (outputs)

def main():
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

	outputs = train(inputs, outputs, synaptic_weights, 1)
	
	print("Outputs after training: ")
	print(outputs)

if __name__ == "__main__":
	main()