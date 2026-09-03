import numpy as np
import matplotlib.pyplot as plt

# This is an already implemented utility function, you do not need to touch it
def plot_regression(x, y, y_pred=None):
    # Expect 1D arrays so each index i pairs x[i] with y[i] (and later y_pred[i]).
    assert len(x.shape) == 1, f"x shape should be ({x.shape[0]},) but it is {x.shape}"
    assert len(y.shape) == 1, f"y shape should be ({y.shape[0]},) but it is {y.shape}"

    plt.scatter(x, y)

    if y_pred is not None:
        # y_pred = model predictions f(x) at these x values — the red line is that fitted function
        # (drawn at discrete x; use many x if you want a smoother curve).
        assert len(y_pred.shape) == 1, f"y_pred shape should be ({y_pred.shape[0]},) but it is {y_pred.shape}"
        # Stack (x, y_pred) as rows so we can sort by x before drawing the fit.
        x_y = np.concatenate((x[:, np.newaxis], y_pred[:, np.newaxis]), axis=-1)
        # Sort by x so plt.plot connects points left-to-right (no zig-zag if x was shuffled).
        x_y = x_y[x_y[:, 0].argsort()]

        plt.plot(x_y[:, 0], x_y[:, 1], color="r")
    plt.show()
