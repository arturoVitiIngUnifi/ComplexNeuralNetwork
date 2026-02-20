import numpy as np
import pandas as pd

n = 10000
K1 = 3
K2 = 8

# Numeri complessi sulla circonferenza unitaria
theta1 = np.random.uniform(0, 2*np.pi, n)
theta2 = np.random.uniform(0, 2*np.pi, n)

x1 = np.exp(1j*theta1)
x2 = np.exp(1j*theta2)

y = K1*x1 + K2*x2

# Funzione per convertire in stringa senza parentesi
def compl_to_str(c):
    return f"{c.real:+.6f}{c.imag:+.6f}j"

df = pd.DataFrame({
    'x1': [compl_to_str(c) for c in x1],
    'x2': [compl_to_str(c) for c in x2],
    'y': [compl_to_str(c) for c in y]
})

df.to_csv("dataset_complesso.csv", index=False)

print(df.head())
