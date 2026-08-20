from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np
import pandas as pd

from mc_quadrants.matrix import nearest_correlation
from mc_quadrants.types import MNTSParameters, RegimeMoments

DEFAULT_TAIL_INDEX = 1.50
DEFAULT_TEMPERING = 0.50


def _sinc(value: float) -> float:
    if abs(value) < 1e-6:
        square = value * value
        return 1.0 - square / 6.0 + square * square / 120.0
    return float(np.sin(value) / value)


def _zolotarev_b_ratio(value: float, alpha: float) -> float:
    complement = 1.0 - alpha
    log_ratio = (
        np.log(max(_sinc(value), 1e-300))
        - alpha * np.log(max(_sinc(alpha * value), 1e-300))
        - complement * np.log(max(_sinc(complement * value), 1e-300))
    )
    return float(np.exp(log_ratio))


def _zolotarev_a(value: float, alpha: float, b_ratio: float) -> float:
    complement = 1.0 - alpha
    log_b_zero = -alpha * np.log(alpha) - complement * np.log(complement)
    return float(np.exp(-(log_b_zero + np.log(max(b_ratio, 1e-300))) / complement))


def _exponentially_tilted_stable(
    rng: np.random.Generator,
    alpha: float,
    tilt: float,
) -> float:
    """Reference implementation of Devroye's exact tilted-stable sampler.

    The C++ implementation is the production path. This scalar version keeps
    the simulator correct and testable on systems where the native extension
    has deliberately been disabled.
    """

    sqrt_pi_over_two = float(np.sqrt(np.pi / 2.0))
    complement = 1.0 - alpha
    tilt_alpha = tilt**alpha
    gamma_parameter = tilt_alpha * alpha * complement
    sqrt_gamma = float(np.sqrt(gamma_parameter))
    xi = (2.0 + sqrt_pi_over_two) * np.sqrt(2.0 * gamma_parameter + 1.0) / np.pi
    psi = (
        np.exp(-gamma_parameter * np.pi * np.pi / 8.0)
        * (2.0 + sqrt_pi_over_two)
        * np.sqrt(gamma_parameter * np.pi)
        / np.pi
    )
    w1 = xi * np.sqrt(np.pi / (2.0 * gamma_parameter))
    w2 = 2.0 * psi * np.sqrt(np.pi)
    w3 = xi * np.pi
    exponent = complement / alpha

    while True:
        while True:
            selector = rng.random()
            mixture_uniform = rng.random()
            if gamma_parameter >= 1.0:
                if selector < w1 / (w1 + w2):
                    angle = abs(rng.standard_normal()) / sqrt_gamma
                else:
                    angle = np.pi * (1.0 - mixture_uniform * mixture_uniform)
            elif selector < w3 / (w3 + w2):
                angle = np.pi * mixture_uniform
            else:
                angle = np.pi * (1.0 - mixture_uniform * mixture_uniform)
            if not 0.0 < angle < np.pi:
                continue

            b_ratio = _zolotarev_b_ratio(float(angle), alpha)
            zeta = np.sqrt(max(b_ratio, 1e-300))
            phi = (sqrt_gamma + alpha * zeta) ** (1.0 / alpha)
            gamma_power = sqrt_gamma ** (1.0 / alpha)
            z_value = phi / max(phi - gamma_power, 1e-300)
            envelope = psi / np.sqrt(max(np.pi - angle, 1e-300))
            envelope += (
                xi * np.exp(-gamma_parameter * angle * angle / 2.0)
                if gamma_parameter >= 1.0
                else xi
            )
            rho_denominator = (1.0 + sqrt_pi_over_two) * sqrt_gamma / zeta + z_value
            log_rho = (
                np.log(np.pi)
                - tilt_alpha * (1.0 - 1.0 / (zeta * zeta))
                + np.log(max(envelope, 1e-300))
                - np.log(max(rho_denominator, 1e-300))
            )
            log_uniform = np.log(max(rng.random(), np.finfo(float).tiny))
            if log_uniform + log_rho > 0.0:
                continue
            acceptance_uniform = np.exp(log_uniform + log_rho)
            a_value = _zolotarev_a(float(angle), alpha, b_ratio)
            break

        mode = (exponent * tilt / a_value) ** alpha
        delta = np.sqrt(mode * alpha / a_value)
        component_1 = delta * sqrt_pi_over_two
        component_2 = delta
        component_3 = z_value / a_value
        component_sum = component_1 + component_2 + component_3
        selector = rng.random()
        normal_draw = 0.0
        exponential_draw = 0.0
        if selector < component_1 / component_sum:
            normal_draw = rng.standard_normal()
            proposal = mode - delta * abs(normal_draw)
        elif selector < (component_1 + component_2) / component_sum:
            proposal = mode + delta * rng.random()
        else:
            exponential_draw = rng.exponential()
            proposal = mode + delta + exponential_draw * component_3
        if proposal < 0.0 or not np.isfinite(proposal):
            continue

        energy = -np.log(max(acceptance_uniform, np.finfo(float).tiny))
        acceptance = a_value * (proposal - mode) + tilt * (
            proposal ** (-exponent) - mode ** (-exponent)
        )
        if proposal < mode:
            acceptance -= normal_draw * normal_draw / 2.0
        if proposal > mode + delta:
            acceptance -= exponential_draw
        if acceptance <= energy:
            return float(proposal ** (-exponent))


def sample_mnts_subordinators(
    rng: np.random.Generator,
    samples: int,
    tail_index: float,
    tempering: float,
) -> np.ndarray:
    """Draw unit-mean tempered-stable subordinators for standardized MNTS."""

    alpha = float(tail_index) / 2.0
    log_scale = ((1.0 - alpha) * np.log(tempering) - np.log(alpha)) / alpha
    scale = float(np.exp(log_scale))
    tilt = float(tempering * scale)
    return scale * np.fromiter(
        (_exponentially_tilted_stable(rng, alpha, tilt) for _ in range(samples)),
        dtype=float,
        count=samples,
    )


def _standardized_shape(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 4:
        dimension = values.shape[1] if values.ndim == 2 else 0
        return np.zeros(dimension), np.full(dimension, 1.5)
    centered = values - values.mean(axis=0)
    scale = np.sqrt(np.mean(centered * centered, axis=0))
    scale = np.maximum(scale, 1e-12)
    standardized = centered / scale
    skewness = np.mean(standardized**3, axis=0)
    excess_kurtosis = np.mean(standardized**4, axis=0) - 3.0
    return (
        np.clip(np.nan_to_num(skewness), -3.0, 3.0),
        np.clip(np.nan_to_num(excess_kurtosis), 0.05, 25.0),
    )


def _fit_common_tail_parameters(
    target_skewness: np.ndarray,
    target_excess_kurtosis: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    best: tuple[float, float, float, np.ndarray] | None = None
    alphas = np.linspace(0.55, 0.95, 17)
    thetas = np.exp(np.linspace(np.log(0.04), np.log(20.0), 65))
    for alpha in alphas:
        variance = ((1.0 - alpha) / thetas)[:, None]
        third = ((1.0 - alpha) * (2.0 - alpha) / thetas**2)[:, None]
        fourth = (
            (1.0 - alpha)
            * (2.0 - alpha)
            * (3.0 - alpha)
            / thetas**3
        )[:, None]
        bound = 0.985 / np.sqrt(variance)
        lower = -np.broadcast_to(bound, (len(thetas), len(target_skewness))).copy()
        upper = np.broadcast_to(bound, lower.shape).copy()

        def skew(values: np.ndarray) -> np.ndarray:
            gamma_squared = np.maximum(0.0, 1.0 - values * values * variance)
            return values**3 * third + 3.0 * values * gamma_squared * variance

        target = np.broadcast_to(target_skewness, lower.shape)
        clipped_target = np.clip(target, skew(lower), skew(upper))
        for _ in range(48):
            middle = 0.5 * (lower + upper)
            below = skew(middle) < clipped_target
            lower = np.where(below, middle, lower)
            upper = np.where(below, upper, middle)
        nus = 0.5 * (lower + upper)
        fitted_skewness = skew(nus)
        gamma_squared = np.maximum(0.0, 1.0 - nus * nus * variance)
        fitted_kurtosis = (
            nus**4 * fourth
            + 6.0 * nus * nus * gamma_squared * third
            + 3.0 * gamma_squared * gamma_squared * variance
        )
        skew_error = np.square(fitted_skewness - target_skewness) / (
            1.0 + np.square(target_skewness)
        )
        kurtosis_error = np.square(fitted_kurtosis - target_excess_kurtosis) / (
            1.0 + np.square(target_excess_kurtosis)
        )
        losses = np.mean(skew_error + kurtosis_error, axis=1)
        losses += 2e-4 * ((alpha - 0.75) / 0.20) ** 2
        theta_index = int(np.argmin(losses))
        loss = float(losses[theta_index])
        if best is None or loss < best[0]:
            best = (loss, alpha, float(thetas[theta_index]), nus[theta_index].copy())
    assert best is not None
    return 2.0 * best[1], best[2], best[3]


def fit_mnts_parameters(
    moments: RegimeMoments,
    state_returns: pd.DataFrame,
    pooled_returns: pd.DataFrame,
    prior_observations: float = 48.0,
) -> MNTSParameters:
    """Fit parsimonious standardized MNTS parameters by pooled moments.

    Tail index and tempering are common within a quadrant. Asset skewness is
    fitted separately, then the latent Gaussian correlation is reconstructed
    so the standardized MNTS covariance matches the calibrated correlation.
    """

    assets = list(moments.mean.index)
    pooled = pooled_returns.reindex(columns=assets).dropna().to_numpy(dtype=float)
    local = state_returns.reindex(columns=assets).dropna().to_numpy(dtype=float)
    pooled_skew, pooled_kurtosis = _standardized_shape(pooled)
    local_skew, local_kurtosis = _standardized_shape(local)
    reliability = len(local) / max(len(local) + float(prior_observations), 1.0)
    target_skewness = reliability * local_skew + (1.0 - reliability) * pooled_skew
    target_kurtosis = reliability * local_kurtosis + (1.0 - reliability) * pooled_kurtosis
    tail_index, tempering, nu = _fit_common_tail_parameters(
        target_skewness,
        target_kurtosis,
    )

    alpha = tail_index / 2.0
    variance_t = (1.0 - alpha) / tempering
    gamma = np.sqrt(np.maximum(1.0 - nu * nu * variance_t, 1e-8))
    target_correlation = moments.correlation.reindex(index=assets, columns=assets)
    latent = (
        target_correlation.to_numpy(dtype=float) - variance_t * np.outer(nu, nu)
    ) / np.outer(gamma, gamma)
    latent_frame = nearest_correlation(
        pd.DataFrame(latent, index=assets, columns=assets)
    )
    return MNTSParameters(
        tail_index=float(tail_index),
        tempering=float(tempering),
        skewness=pd.Series(nu, index=assets, dtype=float),
        gaussian_correlation=latent_frame,
    )


def attach_mnts_parameters(
    moments: Mapping[str, RegimeMoments],
    historical_returns: Mapping[str, pd.DataFrame],
) -> dict[str, RegimeMoments]:
    available = [frame for frame in historical_returns.values() if frame is not None and not frame.empty]
    if not available:
        return dict(moments)
    pooled = pd.concat(available).sort_index()
    calibrated: dict[str, RegimeMoments] = {}
    for state, state_moments in moments.items():
        local = historical_returns.get(state, pooled)
        parameters = fit_mnts_parameters(state_moments, local, pooled)
        calibrated[state] = replace(state_moments, mnts=parameters)
    return calibrated


def resolved_mnts_parameters(moments: RegimeMoments) -> MNTSParameters:
    if moments.mnts is not None:
        return moments.mnts
    assets = list(moments.mean.index)
    return MNTSParameters(
        tail_index=DEFAULT_TAIL_INDEX,
        tempering=DEFAULT_TEMPERING,
        skewness=pd.Series(np.zeros(len(assets)), index=assets, dtype=float),
        gaussian_correlation=moments.correlation.reindex(index=assets, columns=assets),
    )
