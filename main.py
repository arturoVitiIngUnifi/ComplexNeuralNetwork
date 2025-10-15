from network.ComplexNetwork import ComplexNetwork
import random

# 3 input, 8 hidden layer, 2 output
structure = [3, 5, 6, 4, 7, 3, 8, 5, 6, 2]
net = ComplexNetwork(structure)

inputs = [complex(random.random(), random.random()) for _ in range(structure[0])]
print(f"Input Starts: {inputs}")

final_output = net.feedforward(inputs)
print("\n End output:")
for i, y in enumerate(final_output):
    print(f" Neuron Complex output {i}: {y}")