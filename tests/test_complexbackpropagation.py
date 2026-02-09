import unittest
from network.ComplexNetwork import ComplexNetwork

class TestComplexBackpropagationExact(unittest.TestCase):

    def setUp(self):
        # 2 input → 2 hidden → 1 output
        self.net = ComplexNetwork([2, 2, 1], learningRate=0.1)

        self.inputs = [1 + 0j, 0 + 1j]
        self.targets = [1 + 1j]

        # ----- Hidden layer -----
        hidden_layer = self.net.layers[0]

        # weights = [bias, w1, w2]
        hidden_layer[0].weights = [
            0.0 + 0.0j,
            0.1 + 0.0j,
            0.2 + 0.0j
        ]

        hidden_layer[1].weights = [
            0.0 + 0.0j,
            0.3 + 0.0j,
            0.4 + 0.0j
        ]

        # ----- Output layer -----
        output_neuron = self.net.layers[1][0]
        output_neuron.weights = [
            0.0 + 0.0j,
            0.5 + 0.0j,
            0.6 + 0.0j
        ]

    def test_backpropagation_step_exact(self):

        initial_outputs = self.net.feedforward(self.inputs)
        initial_output = initial_outputs[0]

        old_weights = self.net.layers[1][0].weights.copy()
        self.net.backpropagation(self.inputs, self.targets)

        new_outputs = self.net.feedforward(self.inputs)
        new_output = new_outputs[0]

        # Correctness of output direction
        delta_real = self.targets[0].real - initial_output.real
        delta_imag = self.targets[0].imag - initial_output.imag

        self.assertTrue(
            (new_output.real - initial_output.real) * delta_real > 0,
            "Parte reale non aggiornata nella direzione corretta"
        )

        self.assertTrue(
            (new_output.imag - initial_output.imag) * delta_imag > 0,
            "Parte immaginaria non aggiornata nella direzione corretta"
        )

        new_weights = self.net.layers[1][0].weights

        for old_w, new_w in zip(old_weights, new_weights):
            self.assertNotAlmostEqual(
                old_w, new_w, msg="Un peso dell’output layer non è cambiato"
            )


if __name__ == "__main__":
    unittest.main()
