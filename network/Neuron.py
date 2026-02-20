import math
from typing import List, Optional
import random


def activation(z: complex) -> complex:
    """
        Split-complex tanh activation.
        :param z: Linear combination of input with weights.
        :return: Result of activation function
    """
    real_part = math.tanh(z.real)
    imag_part = math.tanh(z.imag)
    return complex(real_part, imag_part)

def linearActivation(z: complex) -> complex:
    """
        :param z: Combination of input with weights from Hidden Layer
        :return: Linear activation for output layer
    """
    return z

class Neuron:
    def __init__( self, inputs: List[complex], weights: Optional[List[complex]] = None,
                  layerIndex: int = 0, isOutputLayer: bool = False, learningRate: float = 0.01 ):

        if layerIndex < 0:
            raise ValueError('Neuron layerIndex must be >= 0')

        self.inputs: List[complex] = inputs
        self.layerIndex: int = layerIndex
        self.isOutputLayer: bool = isOutputLayer
        self.learningRate: float = learningRate

        # Store last output and z for backprop
        self.last_output: complex = 0j
        self.last_z: complex = 0j

        # If weights are not set, initialize them with random complex numbers
        # Xavier initialization scaled for complex numbers
        if weights is None:
            n_weights = len(inputs) + 1  # +1 for bias
            scale = math.sqrt(2.0 / (len(inputs) + 1)) if len(inputs) > 0 else 0.5
            self.weights = [
                complex(
                    random.gauss(0, scale),
                    random.gauss(0, scale)
                )
                for _ in range(n_weights)
            ]
        else:
            self.weights = weights


    def getLinearCombination(self) -> complex:
        """
            :return: Computed weighted sum including bias
        """
        z = self.weights[0]  # bias
        for i, x in enumerate(self.inputs):
            z += self.weights[i + 1] * x
        return z


    def output(self) -> complex:
        """
            Get Output from Activation Function.
            :return: The Complex result
        """
        self.last_z = self.getLinearCombination()
        if self.isOutputLayer:
            self.last_output = linearActivation(self.last_z)
        else:
            self.last_output = activation(self.last_z)
        return self.last_output


    def setWeights( self, weights: List[complex] ) -> None:
        self.weights = weights
