import os
import pandas as pd
import matplotlib.pyplot as plt

from anomaly import compute_scores


def plot_anomalies(y_true, y_pred, errors, log_likelihoods, anomalies, outfile):
    N = len(y_true)

    plt.figure(figsize=(20, 7))

    ax1 = plt.subplot(411)
    ax1.set_title("Actual Signal")
    ax1.plot(y_true, 'blue')
    y_max = max(y_true) * 1.10
    bar_width = 50
    ax1.bar(range(N), y_max * anomalies['high'], bar_width, color='red',
            alpha=0.5, align='center')
    ax1.set_ylim(0, y_max)

    ax2 = plt.subplot(412, sharex=ax1)
    ax2.set_title("Predicted Signal")
    ax2.plot(y_pred, 'green')

    ax3 = plt.subplot(413, sharex=ax2)
    ax3.set_title("Error")
    ax3.plot(errors, 'orange')

    ax4 = plt.subplot(414, sharex=ax3)
    ax4.set_title("Anomaly Log Likelihood")
    ax4.plot(log_likelihoods, 'red')

    # Plot anomalies
    ax4.bar(range(N), anomalies['high'], bar_width, color='red', alpha=0.5,
            align='center')
    ax4.set_ylim(0, 1)

    plt.xlim(0, len(y_true))
    plt.tight_layout()
    plt.savefig(outfile)


def plot_history(output_file, history):
    plt.figure()
    handles = []
    for metric in history.keys():
        h = history[metric]
        line, = plt.plot(range(len(h)), h, label=metric)
        handles.append(line)
    plt.legend(handles=handles)
    plt.savefig(output_file)


def analyze_and_plot_results(results_csv_path, history_csv_path, show_plots=False,
                             xmin=None, xmax=None):

    # Load data.
    df = pd.read_csv(results_csv_path)
    y_true = df.y_true.values.astype("float64")
    y_test = df.y_test.values.astype("float64")
    y_pred = df.y_pred.values.astype("float64")

    # X-axis min and max values
    if xmin is None:
        xmin = 0
    if xmax is None:
        xmax = len(y_pred)

    # Compute anomaly scores.
    errors, log_likelihoods, anomalies = compute_scores(y_test, y_pred)

    # Clip data
    y_true = y_true[xmin:xmax]
    y_test = y_test[xmin:xmax]
    y_pred = y_pred[xmin:xmax]
    errors = errors[xmin:xmax]
    log_likelihoods = log_likelihoods[xmin:xmax]
    anomalies['high'] = anomalies['high'][xmin:xmax]
    anomalies['medium'] = anomalies['medium'][xmin:xmax]

    # Plot data, predictions, and anomaly scores.
    outfile = results_csv_path[:-4] + '.png'
    plot_anomalies(y_true, y_pred, errors, log_likelihoods, anomalies, outfile)

    # Plot training history.
    df = pd.read_csv(history_csv_path)
    metrics = df.columns.values
    history = {m: df[m].values.astype("float64")
               for m in metrics if m != 'epochs'}
    outfile = history_csv_path[:-4] + '.png'
    plot_history(outfile, history)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    results_dir = 'results'
    results_csv_path = os.path.join(results_dir, 'nyc_taxi.csv')
    history_csv_path = os.path.join(results_dir, 'history.csv')
    analyze_and_plot_results(results_csv_path, history_csv_path, show_plots=True,
                             xmin=8200)
