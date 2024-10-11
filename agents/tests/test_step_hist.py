from __future__ import annotations

import numpy as np

from agents.evaluate.step_hist import (
    bootstrap_steps,
    faction_success,
    fit_nbinom,
    nbinom_log_likelihood,
    pull_step_data,
)


def test_pull_step_data(step_data: str):
    sucessful_paths, min_steps, if_sucess, total_games = pull_step_data(
        step_data[:-5], [0, 100]
    )
    assert total_games == 200
    assert len(sucessful_paths) == 117
    assert len(min_steps) == 117
    assert len(if_sucess) == 200
    assert np.all(sucessful_paths < 100) & np.all(sucessful_paths > 0)
    assert np.all(min_steps < 10) & np.all(min_steps > 0)
    assert if_sucess.dtype == bool
    assert isinstance(total_games, int)
    assert sucessful_paths.dtype == int
    assert min_steps.dtype == int


def test_faction_success():
    half_correct = np.tile([True, False], 100)
    assert faction_success(half_correct)[0] == 0.5
    assert np.isclose(faction_success(half_correct)[1], 0.0353553, rtol=1e-4)
    mostly_correct = np.tile([True, True, True, True, False], 200)
    assert faction_success(mostly_correct)[0] == 0.8
    assert np.isclose(faction_success(mostly_correct)[1], 0.01264911, rtol=1e-4)


def test_bootstrap_steps():
    # This test can in theory fail due to random numbers, but it is very unlikely
    num_steps = np.exp(0.3 + np.arange(1, 101) / 24).astype(int) + np.tile(
        np.arange(1, 6), 20
    )
    assert np.isclose(bootstrap_steps(num_steps)[0][0], 23.5, rtol=2e-1)
    assert np.isclose(bootstrap_steps(num_steps)[0][1], 2.262, rtol=8e-1)
    assert np.isclose(bootstrap_steps(num_steps)[1][1], 1.90, rtol=2e-1)
    assert np.isclose(bootstrap_steps(num_steps)[1][1], 1.87, rtol=8e-1)


def test_nbinom_log_likelihood():
    optimal_steps = np.tile(np.arange(1, 6), 20)
    num_steps = np.exp(0.3 + np.arange(1, 101) / 24).astype(int) + optimal_steps

    assert np.isclose(
        nbinom_log_likelihood((1, 0.2), num_steps, optimal_steps), 618.16, rtol=1e-4
    )


def test_fit_nbinom():
    optimal_steps = np.tile(np.arange(1, 6), 20)
    num_steps = np.exp(0.3 + np.arange(1, 101) / 24).astype(int) + optimal_steps

    assert np.isclose(fit_nbinom(num_steps, optimal_steps)[0][0], 0.7968, rtol=1e-2)
    assert np.isclose(fit_nbinom(num_steps, optimal_steps)[0][1], 0.1145, rtol=5e-2)
    assert np.isclose(fit_nbinom(num_steps, optimal_steps)[1][0], 0.03374, rtol=1e-2)
    assert np.isclose(fit_nbinom(num_steps, optimal_steps)[1][1], 0.0076381, rtol=5e-2)
