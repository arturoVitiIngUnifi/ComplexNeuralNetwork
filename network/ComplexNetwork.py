from network.Neuron import Neuron
from typing import List


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
