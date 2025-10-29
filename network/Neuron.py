from typing import List, Optional
import random


def activation(z: complex) -> complex:
    """
        :param z: Linear combination of input with weights.
        :return: Result of activation function
    """
    if abs(z) == 0:
        return 0j  # Fallback
    return z / abs(z)


class Neuron:
    def __init__( self, inputs: List[complex], weights: Optional[List[complex]] = None,
                  layerIndex: int = 0, isOutputLayer: bool = False, learningRate: float = 0.0 ):

        if layerIndex < 0:
            raise ValueError('Neuron layerIndex must be >= 0')

        self.inputs: List[complex] = inputs
        self.layerIndex: int = layerIndex
        self.isOutputLayer: bool = isOutputLayer
        self.learningRate: float = learningRate

        # If weights are not set, initialize them with random complex numbers
        if weights is None:
            self.weights = [complex(random.random(), random.random()) for _ in inputs]
        else:
            self.weights = weights


    def getLinearCombination(self) -> complex:
        """
            :return: Result of linear combination of inputs with weights.
        """
        return sum(w * x for w, x in zip(self.weights, self.inputs))

    def output(self) -> complex:
        """
            Get Output from Activation Function.
            :return: The Complex result
        """
        z = self.getLinearCombination()
        return activation(z)


    def setWeights( self, weights: List[complex] ) -> None:
        self.weights = weights
