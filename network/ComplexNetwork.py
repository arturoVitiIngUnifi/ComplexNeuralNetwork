from network.Neuron import Neuron
from typing import List, Optional


class ComplexNetwork:
    def __init__( self, structure: List[int]):
        """
            :param structure: list with the number of neuron for layer: es: [3, 4, 2]:
                3 input, 4 hidden neurons, 2 output
        """
        self.structure = structure
        self.layers: List[List[Neuron]] = []

        for i in range(1, len(structure)):
            nInputs = structure[i - 1]
            nNeurons = structure[i]
            layer = [
                Neuron(
                    inputs=[0j] * nInputs,
                    layerIndex=i,
                    isOutputLayer=(i == len(structure) - 1)
                )
                for _ in range(nNeurons)
            ]
            self.layers.append(layer)

    def feedforward( self, input_values: List[complex]) -> List[complex]:
        """
            Propagation of input through the network.
            :param input_values: Complex records from dataset
            :return: Complex records after propagation in output before the training
        """
        currentInputs = input_values

        for layer in self.layers:
            outputs = []
            for neuron in layer:
                neuron.inputs = currentInputs
                outputs.append(neuron.output())

            currentInputs = outputs
        return currentInputs

    def backpropagation( self, inputs: List[complex], targets: List[complex] ):

        ### Epoch Start ###
        # Propagate values
        outputs = self.feedforward(inputs)

        # Calculate errors and update weights on Output Layer
        deltas_output = self._backprop_output_layer(targets)

        # Calculate errors and update weights on Hidden Layers
        self._backprop_hidden_layer(deltas_output, inputs)


    # @TODO: make tests, train() function and PyDoc

    def _backprop_output_layer(self, targets: List[complex]):
        output_layer = self.layers[-1]
        # Previous of last layer
        N_prev = len(self.layers[-2]) if len(self.layers) > 1 else len(output_layer[0].inputs)

        deltas_output = []

        for k, neuron in enumerate(output_layer):
            Y_km = neuron.output()
            D_km = targets[k]
            delta_star = D_km - Y_km
            delta = delta_star / (N_prev + 1)   # Normalized Error
            deltas_output.append(delta)
            alpha_km = neuron.learningRate

            # Update Weights
            for i in range(len(neuron.weights)):
                Y_prev = neuron.inputs[i]       # Out of previous layer is input for last layer
                grad = (alpha_km / (N_prev + 1)) * delta * Y_prev.conjugate()
                neuron.weights[i] += grad

        return deltas_output


    def _backprop_hidden_layer( self, deltas_output: List[complex], inputs: List[complex] ):
        deltas_per_layer: List[Optional[List[complex]]] = [None] * len(self.layers)
        deltas_per_layer[-1] = deltas_output

        for j in reversed(range(len(self.layers) - 1)):
            layer = self.layers[j]
            next_layer = self.layers[j + 1]
            N_prev = len(self.layers[j - 1]) if j > 0 else len(inputs)

            deltas = []
            for k, neuron in enumerate(layer):
                # Update Errors
                Z_kj = neuron.getLinearCombination()
                absZ = abs(Z_kj) if abs(Z_kj) > 0 else 1
                # δ_kj = (1/(N_{j-1}+1)) * Σ δ_{i,j+1} * (W_k^{i,j+1})
                error_sum = 0j
                for i, next_neuron in enumerate(next_layer):
                    error_sum += deltas_per_layer[j + 1][i] * next_neuron.weights[k].conjugate()

                delta_kj = (1 / (N_prev + 1)) * error_sum
                deltas.append(delta_kj)

                # Update Weights
                for i in range(len(neuron.weights)):
                    Y_prev = neuron.inputs[i]
                    grad = (neuron.learningRate / ((N_prev + 1) * absZ)) * delta_kj * Y_prev.conjugate()
                    neuron.weights[i] += grad

            deltas_per_layer[j] = deltas
