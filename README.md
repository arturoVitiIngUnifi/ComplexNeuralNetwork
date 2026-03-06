# Complex-Valued Neural Network — Bachelor's Thesis

## Overview

This repository contains the implementation and documentation for a Bachelor's Thesis project on **Complex-Valued Neural Networks (CVNNs)** in Python.

The project implements and compares two training strategies:

- **Standard Backpropagation** — iterative gradient-based weight update applied to every sample.
- **Batch QR Decomposition** — least-squares weight update for the output layer via QR factorisation, applied once per epoch.

Both methods are evaluated against each other and validated against the relevant scientific literature, using three different test datasets.

---

## Project Structure

```
.
├── network/
│   ├── Neuron.py               # Complex neuron with split activation
│   └── ComplexNetwork.py       # CVNN with backprop and QR training
├── tests/                      # Unit tests
├── main.py                     # Satellite data experiment
├── testMackeyGlass.py          # Mackey-Glass time series experiment
├── testComplexRelation.py      # Complex linear relations experiment
└── requirements.txt
```

---

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Satellite Data
```bash
python main.py
```

### Mackey-Glass Time Series
```bash
python testMackeyGlass.py
```

### Complex Linear Relations
```bash
python testComplexRelation.py
```

---

## Running Tests

```bash
python -m pytest tests/
```

---
