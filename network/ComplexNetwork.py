import random

from network.Neuron import Neuron
from typing import List, Optional


class ComplexNetwork:
    def __init__( self, structure: List[int], learningRate: float = 0.01):
        """
            :param structure: list with the number of neuron for layer: es: [3, 4, 2]:
                3 input, 4 hidden neurons, 2 output
        """
        self.structure = structure
        self.layers: List[List[Neuron]] = []
        self.learningRate = learningRate

        for i in range(1, len(structure)):
            nInputs = structure[i - 1]
            nNeurons = structure[i]
            layer = [
                Neuron(
                    inputs=[0j] * nInputs, layerIndex=i, isOutputLayer=(i == len(structure) - 1),
                    learningRate=learningRate
                )
                for _ in range(nNeurons)
            ]
            self.layers.append(layer)

    def __str__(self):
        """
            Return a human-readable string representation of the neural network structure.
    
            Each layer is labeled as "Input", "Hidden", or "Output", followed by a visual
            representation of its neurons using the symbol "O".
    
            Example:
                For a network defined as [3, 4, 1], the output will be:
    
                    Layer 0 (Input):   O O O
                    Layer 1 (Hidden):  O O O O
                    Layer 2 (Output):  O
    
            :return: A string describing the network layers and their neuron counts.
            :rtype: str
        """
        lines = []
        for i, nNeurons in enumerate(self.structure):
            if i == 0:
                layerType = "Input"
            elif i == len(self.structure) - 1:
                layerType = "Output"
            else:
                layerType = "Hidden"
            neurons = " ".join("O" for _ in range(nNeurons))
            lines.append(f"Layer {i} ({layerType}):  {neurons}")
        return "\n".join(lines)

    def feedforward( self, inputValues: List[complex]) -> List[complex]:
        """
            Propagation of input through the network.
            :param inputValues: Complex records from dataset
            :return: Complex records after propagation in output before the training
        """
        currentInputs = inputValues

        for layer in self.layers:
            outputs = []
            for neuron in layer:
                neuron.inputs = currentInputs
                outputs.append(neuron.output())

            currentInputs = outputs
        return currentInputs


    def train( self, xTrain: List[List[complex]], yTrain: List[List[complex]], 
               epochs: int = 100, errorThresholdToStop: float = 1e-4, verbose: bool = True
    ):
        """
            Train the complex-valued neural network using the backpropagation algorithm.

            The training process iteratively propagates all training samples through the
            network (feedforward phase) and adjusts the weights (backpropagation phase)
            to minimize the mean squared error between predicted and target outputs.

            Training stops when either:
              - the number of epochs reaches the specified limit, or
              - the mean error per epoch falls below the given threshold.

            :param xTrain: List of complex-valued input samples (each sample is a list of complex numbers).
            :type xTrain: List[List[complex]]
            :param yTrain: List of complex-valued target outputs corresponding to each input sample.
            :type yTrain: List[List[complex]]
            :param epochs: Maximum number of training iterations (default: 10).
            :type epochs: int
            :param errorThresholdToStop: Mean error threshold for early stopping (default: 1e-3).
            :type errorThresholdToStop: float
            :param verbose: Indicate if log training.
            :type verbose: bool
            :return: List of mean errors per epoch, representing the training history.
            :rtype: List[float]

            Example:
                net=ComplexNetwork([3,4,2])

                history = net.train(xTrain, yTrain, epochs=100, errorThresholdToStop=1e-4)
        """
        M = len(xTrain)
        history = []

        for epoch in range(epochs):
            totalError = 0.0

            indices = list(range(M))
            random.shuffle(indices)

            for r in range(M):
                inputs = xTrain[r]
                targets = yTrain[r]

                outputs = self.feedforward(inputs)
                self.backpropagation(inputs, targets)

                sampleError = sum(abs(t - o) ** 2 for t, o in zip(targets, outputs)) / len(outputs)
                totalError += sampleError

            meanErrorEpoch = totalError / M
            history.append(meanErrorEpoch)

            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1}/{epochs} — Mean Error: {meanErrorEpoch:.6f}")

            if meanErrorEpoch < errorThresholdToStop:
                if verbose:
                    print(f"✅ Training stopped: Error threshold reached at epoch {epoch + 1}")
                break

        return history


    def backpropagation( self, inputs: List[complex], targets: List[complex] ):
        """
            Perform one complete backpropagation step for a single training sample.
            Backpropagation using REAL-VALUED gradient descent on complex parameters.

            Key insight: Treat each complex weight w = a + ib as two real parameters (a, b).
            The gradient is computed using the chain rule on the real MSE loss.

            This method executes both phases of the learning process for one sample:
              1. Feedforward — propagate the input through the network to compute outputs.
              2. Backward pass — compute output errors and update weights in both the
                 output and hidden layers.

            It does not perform multiple epochs or aggregate errors — that is handled
            by the `train()` method.

            :param inputs: Complex-valued input vector for one training sample.
            :type inputs: List[complex]
            :param targets: Complex-valued expected output vector for the same sample.
            :type targets: List[complex]
            :return: None
        """
        # Calculate errors and update weights on Output Layer
        deltasOutput = self._backprop_output_layer(targets)

        # Calculate errors and update weights on Hidden Layers
        if len(self.layers) > 1:
            self._backprop_hidden_layer(deltasOutput, inputs)



    def _backprop_output_layer(self, targets: List[complex]):
        """
            Perform the backpropagation step for the output layer.

            This method computes the output deltas (errors) as the difference between
            the desired target values and the actual outputs produced by the network.
            It then updates the weights of the output neurons according to a normalized
            complex-valued gradient descent rule.

            Mathematical formulation:
                δ_km = (D_km - Y_km) / (N_{m-1} + 1)
                ΔW = α / (N_{m-1} + 1) * δ_km * conj(Y_prev)

            Where:
                - δ_km: delta for output neuron k in the last layer m
                - D_km: desired (target) output for neuron k
                - Y_km: actual output of neuron k
                - N_{m-1}: number of neurons in the previous layer
                - α: learning rate of the neuron
                - Y_prev: output from the previous layer (input to this neuron)

            :param targets: Complex target values corresponding to the expected outputs.
            :type targets: List[complex]
            :return: List of complex deltas computed for the output layer neurons.
            :rtype: List[complex]
        """
        output_layer = self.layers[-1]
        deltasOutput = []

        for k, neuron in enumerate(output_layer):
            output = neuron.last_output
            target = targets[k]

            # Error signal
            error = target - output
            deltasOutput.append(error)

            # Gradient for bias
            grad_bias = error
            neuron.weights[0] += neuron.learningRate * grad_bias

            # Gradient for input weights
            for i in range(len(neuron.inputs)):
                grad = error * neuron.inputs[i].conjugate()
                neuron.weights[i + 1] += neuron.learningRate * grad

        return deltasOutput


    def _backprop_hidden_layer( self, deltasOutput: List[complex], inputs: List[complex] ):
        """
            Perform the backpropagation step for all hidden layers of the network.

            This method computes the error terms (deltas) for each hidden neuron
            based on the deltas from the next layer and updates the corresponding
            neuron weights using a normalized gradient descent rule adapted for
            complex-valued neural networks.

            The process iterates backward from the last hidden layer to the first,
            propagating error corrections layer by layer.

            Hidden layer backpropagation using split-complex chain rule.
            The delta backpropagates as:
            delta_j = sum_k(delta_k * w_jk*) ⊙ f'(z_j)

            :param deltasOutput: List of complex deltas computed for the output layer.
            :type deltasOutput: List[complex]
            :param inputs: List of complex input values from the dataset (used for the first layer).
            :type inputs: List[complex]
            :return: None
        """
        deltas_per_layer: List[Optional[List[complex]]] = [None] * len(self.layers)
        deltas_per_layer[-1] = deltasOutput

        # Backpropagate through hidden layers
        for layer_idx in reversed(range(len(self.layers) - 1)):
            layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]

            deltas = []
            for neuron_idx, neuron in enumerate(layer):
                # Accumulate weighted error from next layer
                weighted_error = 0j
                for next_neuron_idx, next_neuron in enumerate(next_layer):
                    next_delta = deltas_per_layer[layer_idx + 1][next_neuron_idx]
                    # weights[neuron_idx + 1] because weights[0] is bias
                    weight = next_neuron.weights[neuron_idx + 1]
                    weighted_error += next_delta * weight.conjugate()

                # Apply activation derivative
                # For split tanh: f'(z) = (1 - tanh²(Re)) + i*(1 - tanh²(Im))
                output = neuron.last_output
                deriv_real = 1.0 - output.real ** 2
                deriv_imag = 1.0 - output.imag ** 2

                # Split-complex multiplication: (a+ib) * (c+id) = (ac-bd) + i(ad+bc)
                # But for independent components: just multiply each part
                delta_real = weighted_error.real * deriv_real
                delta_imag = weighted_error.imag * deriv_imag
                delta = complex(delta_real, delta_imag)

                deltas.append(delta)

                # Update bias
                neuron.weights[0] += neuron.learningRate * delta

                # Update input weights
                for i in range(len(neuron.inputs)):
                    grad = delta * neuron.inputs[i].conjugate()
                    neuron.weights[i + 1] += neuron.learningRate * grad

            deltas_per_layer[layer_idx] = deltas
