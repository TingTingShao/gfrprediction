
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

# code translated from 
# https://rowannicholls.github.io/python/statistics/hypothesis_testing/passing_bablok.html

class PassingBablok:
    def __init__(self, conf_level=0.95):
        """
        conf_level : float
            Confidence level for the slope/intercept CIs (default 0.95).
        """
        self.conf_level = conf_level
        self.x = None
        self.y = None
        self.S = None
        self.N = None
        self.K = None
        self.b = None
        self.a = None
        self.b_L = None
        self.b_U = None
        self.a_L = None
        self.a_U = None

    def _fit(self, x, y):
        """
        Estimate Passing–Bablok slope/intercept and their CIs.
        
        Parameters
        ----------
        x, y : array-like, shape (n,)
            Paired measurements.
        
        Returns
        -------
        self : fitted estimator
        """
        x = x.values
        y = y.values
        n = len(x)

        # 1) Build list of slopes S_ij
        S = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                xi, xj = x[i], x[j]
                yi, yj = y[i], y[j]
                
                if xi == xj:
                    # vertical → ±∞ depending on y
                    if yi == yj:
                        continue
                    gradient = np.inf if yj > yi else -np.inf
                else:
                    gradient = (yj - yi) / (xj - xi)
                    if gradient == -1:  # per original code
                        continue
                S.append(gradient)

        S = np.sort(np.array(S))
        N = len(S)
        K = np.sum(S < -1)

        # 2) shifted median slope
        if N == 0:
            raise ValueError("No valid pairwise slopes could be calculated.")
        if N % 2 != 0:
            # idx = (N + 1) / 2 + K
            # print(idx)
            # b = S[idx]
            middle_pos = N // 2
            idx = K + middle_pos           # this is an integer
            b   = S[idx]
        else:
            pos1 = N // 2 - 1
            pos2 = N // 2
            idx1 = K + pos1                # ints
            idx2 = K + pos2
            b    = 0.5 * (S[idx1] + S[idx2])

        # 3) Intercept
        a = np.median(y - b * x)

        # 4) Confidence intervals
        C = self.conf_level
        gamma = 1 - C
        z = stats.norm.ppf(1 - gamma / 2)
        n=len(x)
        C_gamma = z * np.sqrt((n * (n - 1) * (2 * n + 5)) / 18)
        M1 = int(np.round((N - C_gamma) / 2))
        M2 = int(N - M1 + 1)

        b_L = S[M1 + K - 1]
        b_U = S[M2 + K - 1]
        a_L = np.median(y - b_U * x)
        a_U = np.median(y - b_L * x)

        # print(b)
        # 5) Store results
        self.x, self.y = x, y
        self.S, self.N, self.K = S, N, K
        self.b, self.a = b, a
        self.b_L, self.b_U = b_L, b_U
        self.a_L, self.a_U = a_L, a_U

        return self

    def predict(self, x):
        """Predict y = a + b*x."""
        return self.a + self.b * np.asarray(x)

    def plot(self, x=None, y=None, ax=None, path=None, ylabel=None, format='eps'):
        """
        Scatter + reference line + regression line + CI bands.
        
        Parameters
        ----------
        x, y : array-like, optional
            If provided, will scatter these instead of self.x/self.y.
        ax : matplotlib Axes, optional
            If None, creates a new figure + axes.
        
        Returns
        -------
        ax : the matplotlib Axes
        """
        if ax is None:
            fig, ax = plt.subplots()
        self._fit(x, y)
        x_data = np.asarray(x) if x is not None else self.x
        y_data = np.asarray(y) if y is not None else self.y
        
        # scatter
        ax.scatter(x_data, y_data,
                   c='k', s=10, alpha=0.3, marker='o')

        left, right = ax.get_xlim()
        bottom, top = ax.get_ylim()

        # left, right=[0, 250]
        # bottom, top=[0, 250]
        # make square

        ax.set_xlim(0 if left < 0 else left, right)
        ax.set_ylim(0 if bottom < 0 else bottom, top)

        # reference y = x
        ax.plot([left, right], [left, right],
                c='grey', ls='--', label='Reference line')

        # regression
        xs = np.array([left, right])
        ys = self.a + self.b * xs
        ax.plot(xs, ys,
                label=f'Regression: {self.b:.2f}x + {self.a:.2f}',
                lw=2)

        # confidence bands
        y_L = self.a_L + self.b_L * xs
        y_U = self.a_U + self.b_U * xs
        ax.plot(xs, y_U, c='tab:blue', alpha=0.4,
                label=f'Upper CI: {self.b_U:.2f}x + {self.a_U:.2f}')
        ax.plot(xs, y_L, c='tab:blue', alpha=0.4,
                label=f'Lower CI: {self.b_L:.2f}x + {self.a_L:.2f}')
        ax.fill_between(xs, y_L, y_U, color='tab:blue', alpha=0.2)

        ax.set_aspect('equal')
        ax.set_title('Passing-Bablok Regression')
        ax.set_xlabel('mGFR mL/min/1.73m2')
        ax.set_ylabel(f"{ylabel} mL/min/1.73m2" if ylabel is not None else 'Predictions mL/min/1.73m2')
        ax.legend(frameon=False)
        ax.set_rasterized(True)
        if path is not None:
            plt.savefig(path, bbox_inches='tight', format=format, dpi=1200)
            plt.close()
        return ax

