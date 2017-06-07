import os
import numpy as np
import pandas as pd

from lstm import create_train_and_test, create_model
from plot import analyze_and_plot_results

# For reproducibility
np.random.seed(1234)

# Hyper-parameters
sequence_length = 48  # 1 day = 48 * 30m
duplication_ratio = 0.04
epochs = 10
batch_size = 50
split_index = 336  # 7 days = 336 * 30m
split_index += sequence_length  # Pad split index by sequence_length

# LSTM layers
layers = {
    'input': 1,
    'hidden1': 64,
    'hidden2': 256,
    'hidden3': 100,
    'output': 1
}

# Make sure output dir exists
output_dir = 'results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Prep data
input_csv = 'nyc_taxi.csv'  # Data sampled every 30m.
df = pd.read_csv(os.path.join('data', input_csv))

data = df.value.values.astype("float64")
X_train, y_train, X_test, y_test, y_true = create_train_and_test(
    data=data, sequence_length=sequence_length,
    duplication_ratio=duplication_ratio, split_index=split_index)

# Create LSTM model
model = create_model(sequence_length=sequence_length, layers=layers)

# Train model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_split=0.0)

# Save history.
df = pd.DataFrame(data={'epochs': range(epochs),
                        'loss': history.history['loss']})
history_csv_path = os.path.join(output_dir, 'history.csv')
df.to_csv(history_csv_path, index=False)

# Predict values.
print("Predicting...")
y_pred = model.predict(X_test)
print("Reshaping...")
y_pred = np.reshape(y_pred, (y_pred.size,))

# Save results.
print("Saving...")
results_csv_path = os.path.join(output_dir, input_csv)
df = pd.DataFrame(data={'y_true': y_true, 'y_test': y_test, 'y_pred': y_pred})
df.to_csv(results_csv_path, index=False)

# Extract anomalies from predictions and plot results.
print("Plotting...")
analyze_and_plot_results(results_csv_path, history_csv_path)
print("Done! Results saved in:", output_dir)