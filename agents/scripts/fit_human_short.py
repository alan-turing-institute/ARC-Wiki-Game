"""Fit the human games using only the short ones.

Any game more than CUTOFF steps above optimal is treated as unknown. The negative
binomial is fitted to the successes within the cutoff, and the total success rate is
the seen success rate divided by the fraction of the fit within the cutoff. This is
not limited to 100%, to check if the fit still gives a reasonable answer.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numdifftools as nd
import numpy as np
import pandas as pd
from scipy.linalg import inv
from scipy.stats import nbinom

from agents.evaluate import step_hist

CUTOFF = 10


def fit_short(
    seen_excess: np.ndarray[int], total_games: int, fix_n: float | None = None
) -> dict:
    """
    Fit the negative binomial to the steps above optimal of the seen successes, and
    get the total success rate. Its error combines the Poisson error on the number
    of seen successes with the fit error on the fraction within the cutoff.
    """
    no_optimum = np.zeros_like(seen_excess)
    n_result, p_result = step_hist.fit_nbinom(
        seen_excess, no_optimum, truncation=CUTOFF, fix_n=fix_n
    )
    params = np.array([n_result[0], p_result[0]])

    def nll(params: np.ndarray) -> float:
        return step_hist.nbinom_log_likelihood(params, seen_excess, no_optimum, CUTOFF)

    # fit_nbinom only returns the standard deviations, so get the covariance
    if fix_n is None:
        cov_mat = inv(nd.Hessian(nll, base_step=[1e-1, 1e-3])(params))
    else:
        cov_mat = np.diag([0, p_result[1] ** 2])
    grad = nd.Gradient(lambda x: nbinom.logcdf(CUTOFF, *x))(params)

    success_rate = len(seen_excess) / (total_games * nbinom.cdf(CUTOFF, *params))
    success_std = success_rate * np.sqrt(1 / len(seen_excess) + grad @ cov_mat @ grad)

    return {
        "n": n_result,
        "p": p_result,
        "success_rate": (success_rate, success_std),
        "nll": nll(params),
    }


def fit_label(fit_name: str, fit: dict) -> str:
    n_text = f"n = {fit['n'][0]:.2f} ± {fit['n'][1]:.2f}, " if fit["n"][1] else ""
    return (
        f"NB Fit, {fit_name}\n"
        f"{n_text}p = {fit['p'][0]:.1%} ± {fit['p'][1]:.1%}\n"
        f"Success rate = {fit['success_rate'][0]:.1%} ± {fit['success_rate'][1]:.1%}\n"
        f"NLL = {fit['nll']:.2f}"
    )


def main() -> None:
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    games_df = pd.read_csv(
        os.path.join(main_dir, "data/human/en_wiki_step_list_human_0.csv"), sep="\t"
    )
    total_games = len(games_df)
    successes = games_df[games_df["found_target"]]
    excess = (successes["num_steps"] - successes["optimal_steps"]).values.astype(int)
    seen, unseen = excess[excess <= CUTOFF], excess[excess > CUTOFF]

    fits = {
        "n free": fit_short(seen, total_games),
        "n = 1": fit_short(seen, total_games, fix_n=1),
    }

    # Same style as fit_distribution.py, but the y axis is the fraction of all games
    # so the area under each fit is its success rate
    x_p = np.arange(0, 101, 1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(
        [seen, unseen],
        bins=np.arange(-1, 101, 1) + 0.5,
        weights=[np.full(len(x), 1 / total_games) for x in (seen, unseen)],
        stacked=True,
        color=["skyblue", "lightgrey"],
        label=["Game Data", "Unknown in fit"],
    )
    ax.axvline(CUTOFF + 0.5, color="grey", linestyle=":", linewidth=1)
    for (fit_name, fit), style in zip(fits.items(), ["k--", "C1-."], strict=True):
        print(fit_label(fit_name, fit), end="\n\n")
        ax.plot(
            x_p,
            fit["success_rate"][0] * nbinom.pmf(x_p, fit["n"][0], fit["p"][0]),
            style,
            label=fit_label(fit_name, fit),
        )

    success_frac = step_hist.faction_success(games_df["found_target"].values)
    ax.set_xlabel("Number of steps above optimal")
    ax.set_ylabel("Fraction of all games")
    ax.set_xlim(0, 60)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax.set_title(f"Human - English Wiki - {total_games:_d} Games")
    ax.legend(
        loc="upper right",
        title=f"Observed success rate = {success_frac[0]:.1%} ± {success_frac[1]:.1%}",
        frameon=False,
    )

    figure_dir = os.path.join(main_dir, "result/figure/human_wiki_summary")
    fig.savefig(
        os.path.join(figure_dir, f"en_wiki_human_short_{CUTOFF}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
