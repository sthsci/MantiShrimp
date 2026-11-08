"""Pure 2-D ABM motility and trajectory summary statistics.

Each Cartesian component follows dmu = -alpha*mu*dt + sigma_mu*dW_mu and
dx = mu*dt + sigma_r*dW_r, with independent standard Wiener processes and no
forces. Here mu is cell polarity, alpha its relaxation rate, and sigma_mu and
sigma_r the polarity and translational noise amplitudes. The speed multiplier
v is fixed at 1. For a constant v != 1, mu here represents v times the original
polarity, and sigma_mu must also be multiplied by v.
The notebook supplies parameter grids, runs sweeps, plots and saves results.
"""

from numbers import Integral, Real

import numpy as np


def _validate_parameters(alpha, sigma_mu, sigma_r):
    if not np.isfinite([alpha, sigma_mu, sigma_r]).all():
        raise ValueError("alpha, sigma_mu and sigma_r must be finite")
    if alpha <= 0 or sigma_mu < 0 or sigma_r < 0:
        raise ValueError("alpha must be positive and noise amplitudes non-negative")


def simulate_ou(alpha, sigma_mu, sigma_r, n_tracks, n_steps, dt, rng,
                initial="stationary"):
    """Return independent 2D positions of shape (n_tracks, n_steps + 1, 2).

    All tracks start at the origin. ``initial='zero'`` matches the ABM's initial
    polarity mu; ``'stationary'`` draws each component from N(0, sigma_mu²/(2 alpha)).
    ``dt`` is the observation interval, not an Euler integration step. ``rng``
    must be a NumPy Generator; its state advances during simulation.
    The Gaussian innovations of mu and its time integral share noise; sampling
    them independently would give the wrong displacement correlations.
    """
    _validate_parameters(alpha, sigma_mu, sigma_r)
    for name, value in (("n_tracks", n_tracks), ("n_steps", n_steps)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if initial not in {"stationary", "zero"}:
        raise ValueError("initial must be 'stationary' or 'zero'")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    z = alpha * dt
    decay = np.exp(-z)
    integrated_decay = -np.expm1(-z) / alpha
    polarity_variance = sigma_mu**2 * (-np.expm1(-2 * z)) / (2 * alpha)
    covariance = sigma_mu**2 * integrated_decay**2 / 2
    # The series avoids cancellation when relaxation is slow relative to a frame.
    if z < 1e-3:
        integral_variance = sigma_mu**2 * dt**3 * (
            1 / 3 - z / 4 + 7 * z**2 / 60 - z**3 / 24 + 31 * z**4 / 2520
        )
    else:
        integral_variance = sigma_mu**2 / alpha**2 * (
            dt - 2 * integrated_decay + (-np.expm1(-2 * z)) / (2 * alpha)
        )
    displacement_variance = integral_variance + sigma_r**2 * dt
    if not np.isfinite([polarity_variance, covariance, displacement_variance]).all():
        raise ValueError("parameters exceed the numerical range for exact sampling")
    polarity_sd = np.sqrt(polarity_variance)
    shared_sd = covariance / polarity_sd if polarity_sd > 0 else 0.0
    independent_sd = np.sqrt(max(0.0, displacement_variance - shared_sd**2))

    mu = np.zeros((n_tracks, 2))
    if initial == "stationary":
        mu = rng.normal(size=(n_tracks, 2)) * sigma_mu / np.sqrt(2 * alpha)
    positions = np.zeros((n_tracks, n_steps + 1, 2))
    for step in range(n_steps):
        first, second = rng.normal(size=(2, n_tracks, 2))
        positions[:, step + 1] = (
            positions[:, step] + integrated_decay * mu
            + shared_sd * first + independent_sd * second
        )
        mu = decay * mu + polarity_sd * first
    return positions


def stationary_msd(lags, alpha, sigma_mu, sigma_r):
    """Analytic 2D ensemble MSD for stationary polarity, in squared length units.

    MSD(t) =
            2 sigma_mu²/alpha² [t - (1-exp(-alpha*t))/alpha]
            + 2 sigma_r² t.
    This expression does not apply to polarity initialized at zero.
    """
    _validate_parameters(alpha, sigma_mu, sigma_r)
    lags = np.asarray(lags, dtype=float)
    if not np.isfinite(lags).all() or np.any(lags < 0):
        raise ValueError("lags must be finite and non-negative")
    z = alpha * lags
    factor = np.empty_like(z)
    small = z < 1e-3
    factor[small] = (
        0.5 - z[small] / 6 + z[small]**2 / 24
        - z[small]**3 / 120 + z[small]**4 / 720
    )
    factor[~small] = (z[~small] + np.expm1(-z[~small])) / z[~small]**2
    return 2 * sigma_mu**2 / alpha * lags**2 * factor + 2 * sigma_r**2 * lags


METRIC_DEFINITIONS = {
    "observed_duration": "Elapsed time from the first to the last finite position.",
    "missing_fraction": "Fraction of frames missing between the first and last finite position, inclusive.",
    "mean_speed": "Mean length of valid adjacent-frame displacements divided by dt.",
    "median_speed": "Median length of valid adjacent-frame displacements divided by dt.",
    "speed_cv": "Population standard deviation of step speed divided by mean step speed.",
    "speed_tail_ratio": "95th percentile of step speed divided by its median; linear quantile interpolation.",
    "mean_turning_angle_deg": "Mean unsigned angle between consecutive nonzero displacement vectors, in degrees.",
    "directional_persistence": "Mean cosine of the angle between consecutive nonzero displacement vectors.",
    "reversal_fraction": "Fraction of consecutive nonzero displacement pairs with an unsigned angle above 90 degrees.",
    "path_length": "Sum of valid adjacent-frame displacement lengths; omitted segments are not bridged.",
    "net_displacement": "Distance between the first and last finite position, including across internal gaps.",
    "straightness": "Net displacement divided by path length; undefined for zero path length or internal gaps.",
    "radius_gyration": "Root mean squared distance of finite positions from their centroid.",
    "msd_exponent": "Unweighted OLS slope of log(time-averaged MSD) against log(lag time), using positive MSD values at lags 1 through max_lag; at least three valid lags required.",
    "msd_first_lag": "Time-averaged squared displacement between finite positions one frame apart.",
    "msd_last_lag": "Time-averaged squared displacement between finite positions max_lag frames apart.",
    "non_gaussian_first_lag": "2-D displacement non-Gaussian parameter <r^4> / (2 <r^2>^2) - 1 at lag one.",
}


TURNING_ANGLE_GRID = np.arange(0.0, 181.0, 5.0)


def _ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0,
    )


def _mean(values, axis=1):
    valid = np.isfinite(values)
    return _ratio(np.where(valid, values, 0.0).sum(axis=axis), valid.sum(axis=axis))


def summarize_tracks(positions, dt, max_lag=15):
    """Return per-track scalar dictionaries and time-averaged curve arrays.

    ``positions`` has shape (tracks, frames, 2); a missing frame has two NaNs.
    Curves ``msd``, ``directional_acf``, ``velocity_acf`` and ``speed_acf`` have
    shape (tracks, max_lag), at lags ``dt * arange(1, max_lag + 1)``. ACF pairs
    must lie within the same contiguous observed segment. Directional ACF uses
    unit displacement vectors, velocity ACF is divided by the track's mean
    squared step speed, and speed ACF uses the track mean and population
    variance. MSD requires finite endpoints but permits intervening gaps.
    ``turning_cdf`` uses the 37 thresholds in ``TURNING_ANGLE_GRID``.

    Speeds and directions refer to observed displacements, not instantaneous
    latent velocities. Zero displacements have no defined direction.
    """
    if isinstance(dt, bool) or not isinstance(dt, Real) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive number")
    if isinstance(max_lag, bool) or not isinstance(max_lag, Integral) or max_lag < 1:
        raise ValueError("max_lag must be a positive integer")
    if np.iscomplexobj(positions):
        raise ValueError("positions must be real")
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 2 or min(positions.shape[:2]) < 1:
        raise ValueError("positions must have shape (tracks >= 1, frames >= 1, 2)")
    if np.isinf(positions).any() or np.any(np.isnan(positions).any(axis=2) != np.isnan(positions).all(axis=2)):
        raise ValueError("each position must contain two finite coordinates or two NaNs")

    ntracks, nframes, _ = positions.shape
    finite = np.isfinite(positions).all(axis=2)
    nfinite = finite.sum(axis=1)
    first = finite.argmax(axis=1)
    last = nframes - 1 - finite[:, ::-1].argmax(axis=1)
    span = np.where(nfinite > 0, last - first + 1, 0)
    duration = np.where(nfinite > 0, (span - 1) * dt, np.nan)
    missing_fraction = 1.0 - _ratio(nfinite, span)

    steps = np.diff(positions, axis=1)
    step_valid = np.isfinite(steps).all(axis=2)
    lengths = np.linalg.norm(steps, axis=2)
    speeds = lengths / dt
    mean_speed = _mean(speeds)
    speed_var = _mean((speeds - mean_speed[:, None]) ** 2)
    quantiles = np.full((2, ntracks), np.nan)
    has_steps = step_valid.any(axis=1)
    if has_steps.any():
        quantiles[:, has_steps] = np.nanquantile(speeds[has_steps], [0.5, 0.95], axis=1)
    path_length = np.where(has_steps, np.nansum(lengths, axis=1), np.nan)
    net_displacement = np.linalg.norm(positions[np.arange(ntracks), last] - positions[np.arange(ntracks), first], axis=1)
    straightness = np.where(missing_fraction == 0, _ratio(net_displacement, path_length), np.nan)
    centroid = _mean(positions, axis=1)
    radius_gyration = np.sqrt(_mean(np.sum((positions - centroid[:, None, :]) ** 2, axis=2)))

    directions = np.divide(steps, lengths[:, :, None], out=np.full_like(steps, np.nan), where=lengths[:, :, None] > 0)
    turn_cos = np.clip(np.sum(directions[:, :-1] * directions[:, 1:], axis=2), -1.0, 1.0)
    turn_angle = np.degrees(np.arccos(turn_cos))
    nturns = np.isfinite(turn_angle).sum(axis=1)
    turning_cdf = np.column_stack([_ratio((turn_angle <= threshold).sum(axis=1), nturns) for threshold in TURNING_ANGLE_GRID])
    curves = {name: np.full((ntracks, max_lag), np.nan) for name in ("msd", "directional_acf", "velocity_acf", "speed_acf")}
    curves["turning_cdf"] = turning_cdf
    # Missing-step prefix counts retain elapsed lag and exclude cross-gap ACF pairs.
    missing_prefix = np.pad(np.cumsum(~step_valid, axis=1), ((0, 0), (1, 0)))
    speed_square_mean = _mean(speeds ** 2)
    centered_speed = speeds - mean_speed[:, None]
    for lag in range(1, min(max_lag, nframes - 1) + 1):
        r2 = np.sum((positions[:, lag:] - positions[:, :-lag]) ** 2, axis=2)
        curves["msd"][:, lag - 1] = _mean(r2)
        if lag >= nframes - 1:
            continue
        continuous = missing_prefix[:, lag + 1:] == missing_prefix[:, :-(lag + 1)]
        dot_direction = np.sum(directions[:, lag:] * directions[:, :-lag], axis=2)
        dot_velocity = np.sum(steps[:, lag:] * steps[:, :-lag], axis=2) / dt ** 2
        dot_speed = centered_speed[:, lag:] * centered_speed[:, :-lag]
        curves["directional_acf"][:, lag - 1] = _mean(np.where(continuous, dot_direction, np.nan))
        curves["velocity_acf"][:, lag - 1] = _ratio(_mean(np.where(continuous, dot_velocity, np.nan)), speed_square_mean)
        curves["speed_acf"][:, lag - 1] = _ratio(_mean(np.where(continuous, dot_speed, np.nan)), speed_var)

    msd = curves["msd"]
    valid_msd = np.isfinite(msd) & (msd > 0)
    log_msd = np.log(np.where(valid_msd, msd, 1.0))
    log_lag = np.log(np.arange(1, max_lag + 1) * dt)
    nfit = valid_msd.sum(axis=1)
    sum_x = np.sum(valid_msd * log_lag, axis=1)
    sum_y = np.sum(log_msd, axis=1)
    slope = _ratio(
        nfit * np.sum(log_msd * log_lag, axis=1) - sum_x * sum_y,
        nfit * np.sum(valid_msd * log_lag ** 2, axis=1) - sum_x ** 2,
    )
    scalars = {
        "observed_duration": duration,
        "missing_fraction": missing_fraction,
        "mean_speed": mean_speed,
        "median_speed": quantiles[0],
        "speed_cv": _ratio(np.sqrt(speed_var), mean_speed),
        "speed_tail_ratio": _ratio(quantiles[1], quantiles[0]),
        "mean_turning_angle_deg": _mean(turn_angle),
        "directional_persistence": _mean(turn_cos),
        "reversal_fraction": _ratio((turn_angle > 90).sum(axis=1), nturns),
        "path_length": path_length,
        "net_displacement": net_displacement,
        "straightness": straightness,
        "radius_gyration": radius_gyration,
        "msd_exponent": np.where(nfit >= 3, slope, np.nan),
        "msd_first_lag": msd[:, 0],
        "msd_last_lag": msd[:, -1],
        "non_gaussian_first_lag": _ratio(_mean(lengths ** 4), 2.0 * _mean(lengths ** 2) ** 2) - 1.0,
    }
    return scalars, curves


def summarize_curves(curves, dt, max_lag=15):
    """Return pooled curve quantiles as records, with lag/angle coordinates."""
    rows = []
    for name, values in curves.items():
        grid = TURNING_ANGLE_GRID if name == "turning_cdf" else dt * np.arange(1, max_lag + 1)
        for i, x in enumerate(grid):
            finite = values[:, i][np.isfinite(values[:, i])]
            quantiles = np.quantile(finite, [0, .05, .5, .95, 1]) if len(finite) else [np.nan] * 5
            rows.append(dict(curve=name, lag_or_angle=x, n_valid=len(finite),
                             **dict(zip(["min", "q05", "median", "q95", "max"], quantiles))))
    return rows


def self_check():
    """Check exact moments, numerical limits, statistic definitions and validation."""
    alpha, sigma_mu, sigma_r, dt = 1.4, 0.8, 0.35, 0.3
    paths = simulate_ou(alpha, sigma_mu, sigma_r, 40_000, 8, dt, np.random.default_rng(137))
    expected = stationary_msd(np.arange(1, 9) * dt, alpha, sigma_mu, sigma_r)
    np.testing.assert_allclose(np.mean(np.sum(paths[:, 1:]**2, axis=2), axis=0), expected, rtol=.025)
    steps = np.diff(paths, axis=1)
    covariance = sigma_mu**2 * (1 - np.exp(-alpha * dt))**2 / (2 * alpha**3)
    np.testing.assert_allclose(np.mean(steps[:, :-1] * steps[:, 1:]), covariance, rtol=.04)
    zero = simulate_ou(alpha, sigma_mu, sigma_r, 40_000, 8, dt,
                       np.random.default_rng(138), initial="zero")
    times = np.arange(1, 9) * dt
    expected_zero = 2 * sigma_mu**2 / alpha**2 * (
        times - 2 * (1 - np.exp(-alpha * times)) / alpha
        + (1 - np.exp(-2 * alpha * times)) / (2 * alpha)
    ) + 2 * sigma_r**2 * times
    np.testing.assert_allclose(np.mean(np.sum(zero[:, 1:]**2, axis=2), axis=0),
                               expected_zero, rtol=.025)
    slow = simulate_ou(1e-8, .4, 0, 30_000, 1, .2, np.random.default_rng(19), initial="zero")
    np.testing.assert_allclose(np.mean(slow[:, -1]**2), .4**2 * .2**3 / 3, rtol=.025)
    first = simulate_ou(1, .4, .2, 4, 7, .2, np.random.default_rng(17))
    np.testing.assert_array_equal(first, simulate_ou(1, .4, .2, 4, 7, .2, np.random.default_rng(17)))
    t = np.arange(21) * .5
    scalars, curves = summarize_tracks(np.stack((3*t, 4*t), axis=-1)[None], .5, 8)
    for key, expected in {
        "mean_speed": 5, "straightness": 1, "msd_exponent": 2,
        "directional_persistence": 1, "reversal_fraction": 0}.items():
        np.testing.assert_allclose(scalars[key], expected, atol=1e-12)
    np.testing.assert_allclose(curves["msd"][0], (5 * np.arange(1, 9) * .5)**2)
    brownian = simulate_ou(2, 0, 1, 256, 500, 1, np.random.default_rng(281))
    scalars, curves = summarize_tracks(brownian, 1, 12)
    np.testing.assert_allclose(np.mean(curves["msd"], axis=0), 2 * np.arange(1, 13), rtol=.04)
    np.testing.assert_allclose(np.mean(scalars["reversal_fraction"]), .5, atol=.015)
    still = simulate_ou(1, 0, 0, 2, 10, 1, np.random.default_rng(1))
    np.testing.assert_array_equal(still, 0)
    scalars, _ = summarize_tracks(still, 1, 3)
    assert np.isnan(scalars["straightness"]).all()
    assert np.isnan(scalars["directional_persistence"]).all()
    for invalid in ({"alpha": 0}, {"sigma_mu": -1}, {"sigma_r": np.inf}, {"dt": 0}):
        values = dict(alpha=1, sigma_mu=1, sigma_r=1, n_tracks=1, n_steps=3, dt=1)
        try:
            simulate_ou(**(values | invalid), rng=np.random.default_rng(1))
        except ValueError:
            continue
        raise AssertionError(f"Accepted invalid input: {invalid}")
    print("PASS: stationary/zero-start OU moments, covariance, slow relaxation, seeds, ballistic/Brownian/static motion and validation")


if __name__ == "__main__":
    self_check()
