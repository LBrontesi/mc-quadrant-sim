"""Experimental dynamic Nelson–Siegel model for Treasury par-yield proxies.

Fixed decay (inverse years); factors follow a ridge VAR with macro predictors
and shrunk regime drift. This is a physical forecasting model, not an
arbitrage-free zero-coupon discount curve.
"""

import numpy as np

CURVE_TENORS = {
    "bond_yield_2y": 2.0,
    "bond_yield_5y": 5.0,
    "bond_yield": 7.0,
    "bond_yield_10y": 10.0,
    "bond_yield_20y": 20.0,
}
DECAY = 0.7308  # Equivalent to 0.0609 when maturities are expressed in months.


def loadings(maturities, decay=DECAY):
    maturity = np.asarray(maturities, dtype=float)
    if not np.isfinite(maturity).all() or (maturity <= 0).any() or not np.isfinite(decay) or decay <= 0:
        raise ValueError("Curve maturities and decay must be finite and positive.")
    x = maturity * decay
    slope = -np.expm1(-x) / x
    return np.stack([np.ones_like(x), slope, slope - np.exp(-x)], axis=-1)


def project_curve(values, specification):
    """Project only nominal Treasury columns; preserve cash, real yields, spreads."""
    indices = specification["indices"]
    basis = np.asarray(specification["loadings"])
    values[..., indices] = values[..., indices] @ np.asarray(specification["inverse"]).T @ basis.T
    return values


def curve_forecast(current, specification, state_weights):
    factors = current[:, specification["indices"]] @ np.asarray(specification["inverse"]).T
    predictors = np.column_stack([factors, current[:, specification["macro_indices"]]])
    standardized = (predictors - specification["mean"]) / specification["scale"]
    change = standardized @ np.asarray(specification["coefficient"])
    change += state_weights @ np.asarray(specification["state_drifts"])
    return (factors + change) @ np.asarray(specification["loadings"]).T


def fit_curve(values, columns, valid, membership, states, macro_columns):
    indices = [columns.index(column) for column in CURVE_TENORS]
    basis = loadings(list(CURVE_TENORS.values()))
    inverse = np.linalg.pinv(basis)
    factors = values[:, indices] @ inverse.T
    macro_indices = [columns.index(column) for column in macro_columns]
    predictors = np.column_stack([factors, values[:, macro_indices]])
    mean = predictors[:-1][valid].mean(axis=0)
    scale = np.maximum(predictors[:-1][valid].std(axis=0), 1e-6)
    x = (predictors[:-1] - mean) / scale
    delta = np.diff(factors, axis=0)
    # Shrink toward a random walk; exclude synthetic bootstrap block joins.
    coefficient = np.linalg.solve(x[valid].T @ x[valid] + 24 * np.eye(x.shape[1]), x[valid].T @ delta[valid])
    # Guard the factor autoregression against explosive extrapolation. This
    # constrains its own block; it is not a no-arbitrage restriction.
    transition = np.eye(3) + coefficient[:3] / scale[:3, None]
    radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if radius > 0.995:
        transition *= 0.995 / radius
        coefficient[:3] = (transition - np.eye(3)) * scale[:3, None]
    residual = delta - x @ coefficient
    global_drift = residual[valid].mean(axis=0)
    drifts = []
    for i in range(len(states)):
        weights = membership[:-1, i] * valid
        drifts.append((weights @ residual + 24 * global_drift) / (weights.sum() + 24))
    specification = dict(
        indices=indices,
        loadings=basis.tolist(),
        inverse=inverse.tolist(),
        macro_indices=macro_indices,
        mean=mean.tolist(),
        scale=scale.tolist(),
        coefficient=coefficient.tolist(),
        state_drifts=np.asarray(drifts).tolist(),
        states=list(states),
        decay=DECAY,
        maturities=list(CURVE_TENORS.values()),
        observations=int(valid.sum()),
        mode="nelson_siegel",
        yield_basis="Treasury par-yield proxy",
    )
    predictions = curve_forecast(values[:-1], specification, membership[:-1])
    return specification, predictions
