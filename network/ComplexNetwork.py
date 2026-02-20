import random
import numpy as np

from typing import List, Optional, Tuple, Dict

from network.Neuron import Neuron, activation, linearActivation


class ComplexNetwork:
    """
        Complex-valued Neural Network with QR Decomposition (Batch Method).
    """

    def __init__(self, structure: List[int], learningRate: float = 0.01,
                 use_qr_batch: bool = True):
        """
            Initialize the complex-valued neural network.

            :param structure: list with the number of neuron for layer: es: [3, 4, 2]:
                    3 input, 4 hidden neurons, 2 output
            :param learningRate: Base learning rate for training
            :param use_qr_batch: If True, use Batch QR method for output layer
                If False, use standard iterative backpropagation
        """
        self.structure = structure
        self.layers: List[List[Neuron]] = []
        self.learningRate = learningRate
        self.use_qr_batch = use_qr_batch

        # Statistics for monitoring QR decomposition
        self.condition_numbers: List[float] = []
        self.qr_applications: int = 0

        # Initialize layers
        for i in range(1, len(structure)):
            nInputs = structure[i - 1]
            nNeurons = structure[i]
            layer = [
                Neuron(
                    inputs=[0j] * nInputs,
                    layerIndex=i,
                    isOutputLayer=(i == len(structure) - 1),
                    learningRate=learningRate
                )
                for _ in range(nNeurons)
            ]
            self.layers.append(layer)

        # Data structures for Batch QR accumulation
        self.batch_data: Dict = {}  # {(layer_idx, neuron_idx): {'A': [], 'delta': []}}
        self.activations: List = []  # Stores outputs of each layer for current sample

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

        if self.use_qr_batch:
            lines.append(f"\nTraining Method: BATCH QR (Output Layer Only)")
            lines.append(f"QR Applications: {self.qr_applications}")
            if self.condition_numbers:
                avg_cond = np.mean(self.condition_numbers)
                lines.append(f"Avg Condition Number: {avg_cond:.2f}")

        return "\n".join(lines)

    def feedforward(self, inputValues: List[complex]) -> List[complex]:
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

    def feedforward_store_z(self, inputValues: List[complex]) -> List[complex]:
        """
            Forward pass that stores Z (weighted sum before activation) and layer outputs.

            Used for Batch QR method to:
            1. Store Z values for error calculation: delta = target - Z
            2. Store layer activations for building matrix A in Batch QR

            :param inputValues: Complex records from dataset
            :return: Complex records after propagation in output before the training
        """
        currentInputs = inputValues
        self.activations = [inputValues]  # Store input layer

        for layer_idx, layer in enumerate(self.layers):
            outputs = []
            for neuron in layer:
                neuron.inputs = currentInputs

                # Calculate Z
                z = neuron.getLinearCombination()
                neuron.last_z = z

                # Calculate output Y
                if neuron.isOutputLayer:
                    output = linearActivation(z)
                else:
                    output = activation(z)

                neuron.last_output = output
                outputs.append(output)

            currentInputs = outputs
            self.activations.append(outputs)

        return currentInputs

    def accumulate_batch_data(self, inputs: List[complex], targets: List[complex]):
        """
            Accumulate data for Batch QR decomposition of output layer.

            According to PDF (pages 21-24):
            - For each neuron k in output layer:
              * Matrix A: M × (N_m-1 + 1) containing [1, O_1, O_2, ..., O_N]
              * Vector delta: M × 1 containing errors δ_k^m* = D_k^m - Z_k^m

            :param inputs: Input sample (not used here, for consistency)
            :param targets: Target output values [D_1^m, D_2^m, ..., D_p^m]
        """
        output_layer_idx = len(self.layers) - 1
        output_layer = self.layers[output_layer_idx]

        # Get inputs to output layer (outputs of previous layer)
        if output_layer_idx == 0:
            layer_inputs = inputs
        else:
            layer_inputs = self.activations[output_layer_idx]

        # For each output neuron, accumulate data
        for neuron_idx, neuron in enumerate(output_layer):
            neuron_key = (output_layer_idx, neuron_idx)

            # Initialize accumulation structures if needed
            if neuron_key not in self.batch_data:
                self.batch_data[neuron_key] = {
                    'A': [],        # List of rows [1, O_1, O_2, ..., O_N]
                    'delta': []     # List of errors
                }

            # Build row of A matrix: [1, O_1, O_2, ..., O_N] (with bias term)
            A_row = [1.0 + 0j] + [o for o in layer_inputs]
            self.batch_data[neuron_key]['A'].append(A_row)

            # Calculate error: delta = target - Z
            # (δ_k^m* = D_k^m - Z_k^m)
            z = neuron.last_z
            target = targets[neuron_idx]
            error = target - z
            self.batch_data[neuron_key]['delta'].append(error)

    def apply_qr_batch_correction(self):
        """
            Apply QR Batch correction to output layer at end of epoch.

            Algorithm (PDF pages 24-27):
            1. For each output neuron k:
               a. Get accumulated matrices: A (M × N_m-1+1), delta (M)
               b. Perform QR decomposition: A = QR
               c. Minimize: ||delta - A∆W||²
               d. Solve triangular system: R_{N_m-1+1} ∆W = [Q^H delta]_{N_m-1+1}

            2. Update weights: W = ∆W (directly set, not accumulated)
        """
        if not self.batch_data:
            return

        output_layer_idx = len(self.layers) - 1
        output_layer = self.layers[output_layer_idx]

        for neuron_idx, neuron in enumerate(output_layer):
            neuron_key = (output_layer_idx, neuron_idx)

            if neuron_key not in self.batch_data:
                continue

            # Convert accumulated lists to numpy arrays
            A = np.array(self.batch_data[neuron_key]['A'], dtype=complex)  # M × (N+1)
            delta = np.array(self.batch_data[neuron_key]['delta'], dtype=complex)  # M

            M, N_plus_1 = A.shape

            # Check if batch is sufficient
            if M < N_plus_1:
                if M < 10:
                    print(f"⚠ Warning: Batch size {M} < number of weights {N_plus_1} for output neuron {neuron_idx}")
                # Fall back to least-squares solution
                dW_solution, _, _, _ = np.linalg.lstsq(A, delta, rcond=None)
                dW = dW_solution
            else:
                # QR Decomposition of input data matrix A
                Q, R = np.linalg.qr(A)

                # Calculate condition number for monitoring
                cond_num = np.linalg.cond(A)
                self.condition_numbers.append(cond_num)

                # Minimize residual ||delta - A∆W||²
                # r² = ||δ - QR∆W||² = ||u||² + ||v||²
                # where u = [Q^T δ]_{N_m-1+1} and v = [Q^T δ]_{M-(N_m-1+1)}
                # Minimum when u = 0: R_{N_m-1+1} ∆W = [Q^T δ]_{N_m-1+1}
                Q_H_delta = Q.conj().T @ delta

                # Extract square part R (N+1 × N+1) and corresponding Q^T δ
                R_square = R[:N_plus_1, :N_plus_1]
                Q_H_delta_square = Q_H_delta[:N_plus_1]

                # Solve triangular system: R * ∆W = Q^T * δ
                try:
                    dW = np.linalg.solve(R_square, Q_H_delta_square)
                except np.linalg.LinAlgError:
                    print(f"⚠ Warning: Singular matrix in QR solve for neuron {neuron_idx}")
                    # Fall back to least-squares
                    dW, _, _, _ = np.linalg.lstsq(A, delta, rcond=None)

            # Update weights with solution
            neuron.weights[0] += self.learningRate * dW[0]  # bias weight
            for i in range(len(neuron.inputs)):
                neuron.weights[i + 1] += self.learningRate * dW[i + 1]

            self.qr_applications += 1

        # Clear batch data for next epoch
        self.batch_data = {}

    def backpropagation_standard(self, inputs: List[complex], targets: List[complex]):
        """
            Standard iterative backpropagation (used when use_qr_batch=False).

            This is the original method, kept for comparison.
            Uses delta = target - output (after activation).
        """
        deltasOutput = self._backprop_output_layer_standard(targets)

        if len(self.layers) > 1:
            self._backprop_hidden_layer(deltasOutput, inputs)

    def _backprop_output_layer_standard(self, targets: List[complex]):
        """
            Standard backpropagation for output layer (iterative method).
            Uses delta = target - Y (after activation).
        """
        output_layer = self.layers[-1]
        deltasOutput = []

        for k, neuron in enumerate(output_layer):
            output = neuron.last_output
            target = targets[k]

            error = target - output
            deltasOutput.append(error)

            grad_bias = error
            neuron.weights[0] += neuron.learningRate * grad_bias

            for i in range(len(neuron.inputs)):
                grad = error * neuron.inputs[i].conjugate()
                neuron.weights[i + 1] += neuron.learningRate * grad

        return deltasOutput


    def _backprop_output_layer_batch(self, targets: List[complex]):
        """
            Backpropagation for output layer (Batch QR method).
            Weights are updated by apply_qr_batch_correction() at end of epoch.

            Calculates deltas for propagation to hidden layers.
            Uses delta = target - Z (before activation).
        """
        output_layer = self.layers[-1]
        deltasOutput = []

        for k, neuron in enumerate(output_layer):
            z = neuron.last_z
            target = targets[k]

            error = target - z
            deltasOutput.append(error)

        return deltasOutput


    def _backprop_hidden_layer(self, deltasOutput: List[complex], inputs: List[complex]):
        """
            Perform backpropagation for all hidden layers.

            Uses standard delta propagation formula.
        """
        deltas_per_layer: List[Optional[List[complex]]] = [None] * len(self.layers)
        deltas_per_layer[-1] = deltasOutput

        for layer_idx in reversed(range(len(self.layers) - 1)):
            layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]

            deltas = []
            for neuron_idx, neuron in enumerate(layer):
                # Calculate weighted error from next layer
                weighted_error = 0j
                for next_neuron_idx, next_neuron in enumerate(next_layer):
                    next_delta = deltas_per_layer[layer_idx + 1][next_neuron_idx]
                    weight = next_neuron.weights[neuron_idx + 1]
                    weighted_error += next_delta * weight.conjugate()

                # Apply activation derivative
                # For split tanh: f'(z) = (1 - tanh²(Re)) + i*(1 - tanh²(Im))
                output = neuron.last_output
                deriv_real = 1.0 - output.real ** 2
                deriv_imag = 1.0 - output.imag ** 2

                delta_real = weighted_error.real * deriv_real
                delta_imag = weighted_error.imag * deriv_imag
                delta = complex(delta_real, delta_imag)

                deltas.append(delta)

                # Update weights (iterative method)
                neuron.weights[0] += neuron.learningRate * delta

                for i in range(len(neuron.inputs)):
                    grad = delta * neuron.inputs[i].conjugate()
                    neuron.weights[i + 1] += neuron.learningRate * grad

            deltas_per_layer[layer_idx] = deltas

    def train( self, xTrain: List[List[complex]], yTrain: List[List[complex]],
              epochs: int = 100, errorThresholdToStop: float = 1e-4,
              verbose: bool = True ) -> List[float]:
        """
            Train the complex-valued neural network.

            Algorithm (for use_qr_batch=True):
            ====================================
            For each epoch:
              1. Accumulate batch data through entire epoch
                 - Forward pass (store Z and activations)
                 - Accumulate matrices A and errors delta
              2. At end of epoch: Apply QR batch correction to output layer
              3. Backpropagate to hidden layers using deltas

            Algorithm (for use_qr_batch=False):
            ====================================
            Standard iterative backpropagation (original method).

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
        """
        M = len(xTrain)
        history = []

        for epoch in range(epochs):
            totalError = 0.0
            indices = list(range(M))
            random.shuffle(indices)

            if self.use_qr_batch:
                # BATCH QR METHOD

                # STEP 1: Accumulate batch data for entire epoch
                self.batch_data = {}

                for r in indices:
                    inputs = xTrain[r]
                    targets = yTrain[r]

                    # Forward pass storing Z values
                    outputs = self.feedforward_store_z(inputs)

                    # Accumulate data for QR batch correction
                    self.accumulate_batch_data(inputs, targets)

                    # Calculate error for monitoring
                    sampleError = sum(abs(t - o) ** 2 for t, o in zip(targets, outputs)) / len(outputs)
                    totalError += sampleError

                # STEP 2: Apply QR Batch Correction
                self.apply_qr_batch_correction()

                # STEP 3: Backpropagate to hidden layers
                if len(self.layers) > 1:
                    for r in indices:
                        inputs = xTrain[r]
                        targets = yTrain[r]

                        # Forward pass
                        self.feedforward_store_z(inputs)

                        # Calculate deltas for output layer
                        deltasOutput = self._backprop_output_layer_batch(targets)

                        # Backprop to hidden layers
                        self._backprop_hidden_layer(deltasOutput, inputs)

            else:
                # Standard Back Propagation
                for r in indices:
                    inputs = xTrain[r]
                    targets = yTrain[r]

                    outputs = self.feedforward(inputs)
                    self.backpropagation_standard(inputs, targets)

                    sampleError = sum(abs(t - o) ** 2 for t, o in zip(targets, outputs)) / len(outputs)
                    totalError += sampleError

            meanErrorEpoch = totalError / M
            history.append(meanErrorEpoch)

            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                cond_info = ""
                if self.use_qr_batch and self.condition_numbers:
                    cond_avg = np.mean(self.condition_numbers[-max(1, len(self.condition_numbers) // 10):])
                    cond_info = f" — Cond Avg: {cond_avg:.2f}"
                print(f"Epoch {epoch + 1}/{epochs} — Mean Error: {meanErrorEpoch:.6f}{cond_info}")

            if meanErrorEpoch < errorThresholdToStop:
                if verbose:
                    print(f"✅ Training stopped: Error threshold reached at epoch {epoch + 1}")
                break

        return history

    def get_statistics(self) -> dict:
        """
            Get QR decomposition statistics.

            Returns dictionary containing:
            - qr_applications: number of times QR was applied (end of epoch)
            - avg_condition_number: average condition number of input matrix A
            - max/min_condition_number: extreme values
            - training_method: "Batch QR" or "Standard Backprop"
        """
        return {
            'qr_applications': self.qr_applications,
            'avg_condition_number': np.mean(self.condition_numbers) if self.condition_numbers else 0,
            'max_condition_number': np.max(self.condition_numbers) if self.condition_numbers else 0,
            'min_condition_number': np.min(self.condition_numbers) if self.condition_numbers else 0,
            'training_method': 'Batch QR (Output Layer)' if self.use_qr_batch else 'Standard Backprop'
        }