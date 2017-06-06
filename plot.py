import matplotlib.pyplot as plt


def plot_results(y_true, y_pred, error, output_file):
    plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(311)
    ax1.set_title("Actual Signal")
    ax1.plot(y_true, 'b')

    ax2 = plt.subplot(312, sharex=ax1)
    ax2.set_title("Predicted Signal")
    ax2.plot(y_pred, 'g')

    ax3 = plt.subplot(313, sharex=ax2)
    ax3.set_title("Error")
    ax3.plot(error, 'r')

    plt.xlim(0, len(y_true))
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
