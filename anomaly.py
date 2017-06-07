import numpy as np
from nupic.algorithms.anomaly_likelihood import AnomalyLikelihood


def compute_scores(y_test, y_pred, normalize=False):
    # Errors
    errors = np.array((y_test - y_pred) ** 2)
    if normalize:
        errors = errors / float(errors.max() - errors.min())

    # Log likelihood.
    log_likelihoods = []
    anomaly_likelihood = AnomalyLikelihood()
    for i in range(len(y_test)):
        likelihood = anomaly_likelihood.anomalyProbability(y_test[i],
                                                           errors[i],
                                                           timestamp=None)
        log_likelihood = anomaly_likelihood.computeLogLikelihood(likelihood)
        log_likelihoods.append(log_likelihood)

    # Anomaly thresholds:
    # - HIGH: log_likelihood >= 0.5
    # - MEDIUM: 0.5 > log_likelihood >= 0.4
    N = len(log_likelihoods)
    anomalies = {'high': np.zeros(N), 'medium': np.zeros(N)}
    x = np.array(log_likelihoods)
    high_idx = x >= 0.5
    anomalies['high'][high_idx] = 1
    # medium_idx = np.logical_and(x >= 0.4, x < 0.5)
    # anomalies['medium'][medium_idx] = 1

    return errors, log_likelihoods, anomalies
