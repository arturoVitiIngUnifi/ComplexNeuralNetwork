import unittest
from network.ComplexNetwork import ComplexNetwork

class TestComplexBackpropagationExact(unittest.TestCase):
    def setUp(self):
        self.net = ComplexNetwork([2, 2, 1])

        self.inputs = [1 + 0j, 0 + 1j]
        self.targets = [1 + 1j]

        hidden_layer = self.net.layers[0]
        hidden_layer[0].weights = [0.1 + 0.0j, 0.2 + 0.0j]
        hidden_layer[1].weights = [0.3 + 0.0j, 0.4 + 0.0j]
        hidden_layer[0].learningRate = 0.1
        hidden_layer[1].learningRate = 0.1

        # Estimated Weights
        output_neuron = self.net.layers[1][0]
        output_neuron.weights = [0.5 + 0.0j, 0.6 + 0.0j]
        output_neuron.learningRate = 0.1

    def test_backpropagation_step_exact(self):
        initial_outputs_hidden = [n.output() for n in self.net.layers[0]]
        initial_output_final = self.net.layers[1][0].output()

        self.net.backpropagation(self.inputs, self.targets)

        new_outputs_hidden = [n.output() for n in self.net.layers[0]]
        new_output_final = self.net.layers[1][0].output()

        # Verify output values
        delta_real = (self.targets[0].real - initial_output_final.real)
        delta_imag = (self.targets[0].imag - initial_output_final.imag)
        self.assertTrue((new_output_final.real - initial_output_final.real) * delta_real > 0,
                        "L'output reale non si è aggiornato nella direzione corretta.")
        self.assertTrue((new_output_final.imag - initial_output_final.imag) * delta_imag > 0,
                        "L'output immaginario non si è aggiornato nella direzione corretta.")

        # Check Weights
        output_neuron = self.net.layers[1][0]
        for old_w, new_w in zip([0.5 + 0.0j, 0.6 + 0.0j], output_neuron.weights):
            self.assertNotAlmostEqual(old_w, new_w, msg="Il peso dell'output neuron non è cambiato.")


if __name__ == "__main__":
    unittest.main()
