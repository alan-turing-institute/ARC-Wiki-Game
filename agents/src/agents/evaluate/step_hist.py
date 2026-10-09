from __future__ import annotations

import numdifftools as nd
import numpy as np
import pandas as pd
from scipy.linalg import inv
from scipy.optimize import minimize
from scipy.stats import bootstrap, nbinom

__all__ = [
    "pull_step_data",
    "faction_success",
    "bootstrap_steps",
    "nbinom_log_likelihood",
    "fit_nbinom",
]


def pull_step_data(
    data_loc: str, number_list: list[int]
) -> tuple[np.ndarray[int], np.ndarray[int], np.ndarray[bool], int]:
    """
    Pull the data from the csv files and return the for the successful paths the path
    length and optimal steps, and for all games if the target was found and the total
    number of games.

    Args:
        data_loc (str): The location of the data files.
        number_list (list[int]): The index of the files to pull.

    Returns:
        sucessful_paths (np.ndarray): The path length for the successful paths.
        min_steps (np.ndarray): The optimal steps for the successful paths.
        if_sucess (np.ndarray): If the target was found.
        total_games (int): The total number of games.
    """
    results_list = []

    for num in number_list:
        results_list.append(
            pd.read_csv(
                data_loc + str(num) + ".csv",
                sep="\t",
            )
        )

    results_df = pd.concat(results_list)
    sucessful_paths = results_df["num_steps"][(results_df["found_target"])].values
    min_steps = results_df["optimal_steps"][(results_df["found_target"])].values
    if_sucess = results_df["found_target"].values
    total_games = len(results_df)

    return sucessful_paths, min_steps, if_sucess, total_games


def faction_success(if_success: np.ndarray[bool]) -> tuple[float, float]:
    """
    Calculate the fraction of successful paths and the standard deviation of the
    fraction.

    Args:
        if_success (np.ndarray[bool]): An array of boolean values indicating if the
            agent found the target. True indicates success.

    Returns:
        p_sucess (float): The fraction of successful paths.
        std (float): The standard deviation of the fraction.
    """
    total = len(if_success)
    p_sucess = if_success.mean()
    return p_sucess, np.sqrt(p_sucess * (1 - p_sucess) / total)


def bootstrap_steps(
    step_count: np.ndarray[int], **bootstrap_kargs
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Generate a bootstrap sample of the step counts.

    Args:
        step_count (np.ndarray): An array of successful path counts.
        bootstrap_kargs: Additional keyword arguments to pass to the
            scipy.stats.bootstrap function.

    Returns:
        mean_result (tuple[float, float]): The mean of the step counts and the standard
            error of the mean.
        std_result (tuple[float, float]): The standard deviation of the step counts and
            the standard error of the standard deviation.
    """
    mean_result = (
        np.mean(step_count),
        bootstrap(step_count[None, :], np.mean, **bootstrap_kargs).standard_error,
    )
    std_result = (
        np.std(step_count),
        bootstrap(step_count[None, :], np.std, **bootstrap_kargs).standard_error,
    )

    return mean_result, std_result


def nbinom_log_likelihood(
    params: tuple[float, float],
    step_count: np.ndarray[int],
    optimum: np.ndarray[int],
    truncation: int = 100,
) -> float:
    """
    Calculate the negative binomial log-likelihood for given parameters and data. This
    uses a slightly modified version as the distribution is truncated.
    Args:
        params (tuple[float, float]): A tuple containing the parameters of the negative
            binomial distribution.
        step_count (np.ndarray): An array of successful path counts.
        optimum (np.ndarray): An array of minimum possible steps corresponding to the
            successful paths.
        truncation (int): The truncation point for the distribution. Defaults to 100.
    Returns:
        float: The log-likelihood value.
    """
    probs = -nbinom.logpmf(step_count - optimum, *params).sum()
    trunc = nbinom.logcdf(truncation - optimum, *params).sum()
    return probs + trunc


def fit_nbinom(
    path_counts: np.ndarray[int],
    optimial_steps: np.ndarray[int],
    truncation: int = 100,
    fix_n: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Fit a negative binomial distribution to the data and return the parameters and
    standard deviation of the parameters.

    Args:
        path_counts (np.ndarray): An array of successful path counts.
        optimial_steps (np.ndarray): An array of minimum possible steps corresponding to
            the successful paths.
        truncation (int): The truncation point for the distribution. Defaults to 100.
        fix_n (float | None): If provided, the n parameter will be fixed to one value.
            Defaults to None.
    Returns:
        n_result (tuple[float, float]): The n parameter of the negative binomial, which
            is the required number of successful trials. This is a tuple containing the
            parameter and the standard deviation.
        p_result (tuple[float, float]): The p parameter of the negative binomial, which
            is the probability of success on each trial. This is a tuple containing the
            parameter and the standard deviation.
    """
    if fix_n is not None:
        nbinom_results = minimize(
            nbinom_log_likelihood,
            x0=[fix_n, 0.5],
            bounds=[(fix_n, fix_n), (0, 1)],
            args=(path_counts, optimial_steps, truncation),
            method="Nelder-Mead",
        )
    else:
        nbinom_results = minimize(
            nbinom_log_likelihood,
            x0=[4, 0.5],
            bounds=[(0, 100), (0, 1)],
            args=(path_counts, optimial_steps, truncation),
            method="Nelder-Mead",
        )

    hessian_func = nd.Hessian(
        lambda params: nbinom_log_likelihood(
            params, path_counts, optimial_steps, truncation=truncation
        ),
        # These next arguments make their way to the step size calculation
        base_step=[1e-1, 1e-3],
        step_ratio=2,
        num_steps=4,
    )
    hessian_mat = hessian_func(nbinom_results.x)
    if np.any(np.isnan(hessian_mat)):
        return (nbinom_results.x[0], np.nan), (nbinom_results.x[1], np.nan)
    if fix_n is not None:
        return (fix_n, 0.0), (nbinom_results.x[1], 1 / np.sqrt(hessian_mat[1, 1]))
    cov_mat = inv(hessian_mat)
    result_std = np.sqrt(np.diag(cov_mat))

    return (nbinom_results.x[0], result_std[0]), (nbinom_results.x[1], result_std[1])
