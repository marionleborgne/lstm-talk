import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from anomaly import compute_scores

# Figure styling
sns.set_style('ticks')
sns.despine()
sns.set_context("talk", rc={'lines.linewidth': 2.0})


def plot_anomalies(timestamps, y_true, y_test, y_pred, errors, log_likelihoods,
                   anomalies, output_file):
    N = len(timestamps)
    y_max = max(y_true) * 1.10
    bar_width = 100

    fig, ax = plt.subplots(figsize=(20, 10), nrows=5, sharex=True)

    # Update labels (1 day = 48 * 30m)
    offset = 8  # 4 hours
    indices = range(offset, N, 2 * 48)
    # indices = range(offset + 3 * 48, N, 14 * 48)
    labels = [timestamps[i].split(' ')[0] for i in indices]
    plt.xticks(indices, labels, rotation=-45, ha='left')

    ax[0].set_title("Actual Signal")
    ax[0].plot(y_true, 'blue')
    ax[0].bar(range(N), y_max * anomalies['high'], bar_width, color='red',
              alpha=0.5, align='center')
    ax[0].set_ylim(0, y_max)

    ax[1].set_title("Normalized Signal")
    ax[1].plot(y_pred, 'purple')

    ax[2].set_title("Predicted Signal")
    ax[2].plot(y_pred, 'green')

    ax[3].set_title("Prediction Error")
    ax[3].plot(errors, 'red')

    ax[4].set_title("Anomaly Log Likelihood")
    ax[4].plot(log_likelihoods, 'black')

    # Plot anomalies
    ax[4].bar(range(N), anomalies['high'], bar_width, color='red', alpha=0.5,
              align='center')
    ax[4].set_ylim(0, 1)

    plt.xlim(0, len(y_true))
    plt.tight_layout()
    plt.savefig(output_file)


def plot_history(output_file, history):
    plt.figure()
    handles = []
    for metric in history.keys():
        h = history[metric]
        line, = plt.plot(range(len(h)), h, label=metric)
        handles.append(line)
    plt.legend(handles=handles)
    plt.savefig(output_file)


def analyze_and_plot_results(results_csv_path, history_csv_path,
                             show_plots=False, xmin=None, xmax=None):
    # Load data.
    df = pd.read_csv(results_csv_path)
    y_true = df.y_true.values.astype("float64")
    y_test = df.y_test.values.astype("float64")
    y_pred = df.y_pred.values.astype("float64")
    timestamps = df.timestamps.values

    # X-axis min and max values
    if xmin is None:
        xmin = 0
    if xmax is None:
        xmax = len(y_pred)

    # Compute anomaly scores.
    errors, log_likelihoods, anomalies = compute_scores(y_test, y_pred)

    # Clip data
    y_true = y_true[xmin:xmax]
    timestamps = timestamps[xmin:xmax]
    y_test = y_test[xmin:xmax]
    y_pred = y_pred[xmin:xmax]
    errors = errors[xmin:xmax]
    log_likelihoods = log_likelihoods[xmin:xmax]
    anomalies['high'] = anomalies['high'][xmin:xmax]
    anomalies['medium'] = anomalies['medium'][xmin:xmax]

    # Plot data, predictions, and anomaly scores.
    output_file = results_csv_path[:-4] + '.png'
    plot_anomalies(timestamps, y_true, y_test, y_pred, errors, log_likelihoods,
                   anomalies, output_file)

    # Plot training history.
    df = pd.read_csv(history_csv_path)
    metrics = df.columns.values
    history = {m: df[m].values.astype("float64")
               for m in metrics if m != 'epochs'}
    output_file = history_csv_path[:-4] + '.png'
    plot_history(output_file, history)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    xmin = 8200
    # xmin = 0

    results_dir = 'results'
    results_csv_path = os.path.join(results_dir, 'nyc_taxi.csv')
    history_csv_path = os.path.join(results_dir, 'history.csv')
    analyze_and_plot_results(results_csv_path, history_csv_path,
                             show_plots=False, xmin=xmin)
