import unittest
import numpy as np

from network.ComplexNetwork import ComplexNetwork


class TestComplexQRDecompositionStep(unittest.TestCase):
    """
        Tests for the Batch QR decomposition step of ComplexNetwork.
        1. Shape of accumulated matrix A
        2. Delta vector contains target - Z for each sample
        3. Weights actually change after correction
        4. batch_data is cleared after apply_qr_batch_correction()
        5. Error decreases after a full training epoch
    """

    def setUp(self):
        """
            2 input → 2 hidden → 1 output network, QR mode.
            Fixed weights for reproducibility.
        """
        self.net = ComplexNetwork([2, 2, 1], learningRate=0.1, use_qr_batch=True)

        # ----- Hidden layer -----
        hidden = self.net.layers[0]
        hidden[0].weights = [0.0 + 0.0j, 0.1 + 0.0j, 0.2 + 0.0j]
        hidden[1].weights = [0.0 + 0.0j, 0.3 + 0.0j, 0.4 + 0.0j]

        # ----- Output layer -----
        out = self.net.layers[1][0]
        out.weights = [0.0 + 0.0j, 0.5 + 0.0j, 0.6 + 0.0j]

        self.samples = [
            ([1 + 0j,  0 + 1j],  [1 + 1j]),
            ([0 + 1j,  1 + 0j],  [0 + 1j]),
            ([1 + 1j,  1 + 1j],  [2 + 0j]),
            ([0 + 0j,  1 + 1j],  [1 + 0j]),
            ([1 + 0j,  1 + 0j],  [0 + 1j]),
        ]

    def _run_forward_and_accumulate(self):
        """Forward pass + accumulate for every sample in self.samples."""
        self.net.batch_data = {}
        for inputs, targets in self.samples:
            self.net.feedforward_store_z(inputs)
            self.net.accumulate_batch_data(inputs, targets)


    # ------------------------------------------------------------------
    # 1. Shape of accumulated matrix A
    # ------------------------------------------------------------------
    def test_matrix_A_shape(self):
        """A must be M × (N_hidden + 1) = 5 × 3 (2 hidden neurons + bias)."""
        self._run_forward_and_accumulate()

        output_layer_idx = len(self.net.layers) - 1
        key = (output_layer_idx, 0)          # only one output neuron

        self.assertIn(key, self.net.batch_data)

        A = np.array(self.net.batch_data[key]['A'], dtype=complex)
        self.assertEqual(A.shape, (5, 3), f"Expected shape (5, 3), got {A.shape}")

    # ------------------------------------------------------------------
    # 2. Delta vector = target - Z  (not target - Y)
    # ------------------------------------------------------------------
    def test_delta_equals_target_minus_Z(self):
        """
            Each entry in delta must equal target - last_z of the output neuron,
            NOT target - last_output (i.e. before activation, not after).
        """
        self._run_forward_and_accumulate()

        key = (len(self.net.layers) - 1, 0)
        deltas = self.net.batch_data[key]['delta']

        # Re-run to collect Z values in order
        self.net.batch_data = {}
        z_values = []
        for inputs, targets in self.samples:
            self.net.feedforward_store_z(inputs)
            z_values.append(self.net.layers[-1][0].last_z)
            self.net.accumulate_batch_data(inputs, targets)

        expected_deltas = [
            self.samples[i][1][0] - z_values[i]
            for i in range(len(self.samples))
        ]

        for i, (d, e) in enumerate(zip(deltas, expected_deltas)):
            self.assertAlmostEqual(d, e, places=10, msg=f"Sample {i}: delta={d} ≠ target-Z={e}")

    # ------------------------------------------------------------------
    # 3. Weights change after apply_qr_batch_correction
    # ------------------------------------------------------------------
    def test_weights_change_after_qr_correction(self):
        """Output neuron weights must be different after QR correction."""
        self._run_forward_and_accumulate()

        old_weights = list(self.net.layers[-1][0].weights)
        self.net.apply_qr_batch_correction()
        new_weights = list(self.net.layers[-1][0].weights)

        changed = any(
            abs(old - new) > 1e-12
            for old, new in zip(old_weights, new_weights)
        )
        self.assertTrue(changed, "No weight changed after QR correction")


    # ------------------------------------------------------------------
    # 4. batch_data cleared after correction
    # ------------------------------------------------------------------
    def test_batch_data_cleared_after_correction(self):
        """batch_data must be empty ({}) after apply_qr_batch_correction()."""
        self._run_forward_and_accumulate()
        self.net.apply_qr_batch_correction()

        self.assertEqual(self.net.batch_data, {},"batch_data not cleared after QR correction")


    # ------------------------------------------------------------------
    # 5. Error decreases after a full training epoch
    # ------------------------------------------------------------------
    def test_error_decreases_after_one_epoch(self):
        """
            After several training epochs with QR, the final mean squared error
            must be strictly lower than before training.
        """
        xTrain = [s[0] for s in self.samples]
        yTrain = [s[1] for s in self.samples]

        def mse():
            return sum(
                abs(self.net.feedforward(x)[0] - y[0]) ** 2
                for x, y in zip(xTrain, yTrain)
            ) / len(xTrain)

        error_before = mse()
        self.net.train(xTrain, yTrain, epochs=20, verbose=False)
        error_after = mse()

        self.assertLess(error_after, error_before,
                        f"Error did not decrease: before={error_before:.6f}  after={error_after:.6f}")


if __name__ == "__main__":
    unittest.main()