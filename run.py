import os
import numpy as np
import pandas as pd

from lstm import create_train_and_test, create_model
from plot import plot_results

# For reproducibility
np.random.seed(1234)

# Hyper-parameters
sequence_length = 100
duplication_ratio = 0.05
epochs = 1
batch_size = 50
split_index = 1000

# LSTM layers
layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100,
          'output': 1}

# Prep data
input_csv = 'nyc_taxi.csv'
df = pd.read_csv(os.path.join('data', input_csv))
data = df.value.values.astype("float64")
X_train, y_train, X_test, y_test = create_train_and_test(
    data=data, sequence_length=sequence_length,
    duplication_ratio=duplication_ratio, split_index=split_index)

# Create LSTM
model = create_model(sequence_length=sequence_length, layers=layers)

# Train model
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
          validation_split=0.0)

# Predict values
print("Predicting...")
y_pred = model.predict(X_test)
print("Reshaping...")
y_pred = np.reshape(y_pred, (y_pred.size,))

# Compute prediction error
mse = ((y_test - y_pred) ** 2)

# Save results
output_dir = 'results'
output_csv = os.path.join(output_dir, input_csv)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.DataFrame(data={'actual': y_test, 'predicted': y_pred, 'error': mse})
df.to_csv(output_csv, index=False)

# Plot anomaly results
output_fig = output_csv[:-4] + '.png'
plot_results(y_test, y_pred, mse, output_fig)
