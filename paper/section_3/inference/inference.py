#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _validate_counts(contacts_per_cell, obs_time: float) -> Tuple[np.ndarray, float]:
    T = float(obs_time)
    if T <= 0:
        raise ValueError("obs_time must be > 0")

    N = np.asarray(contacts_per_cell, dtype=int)
    if N.ndim != 1 or np.any(N < 0):
        raise ValueError("contacts_per_cell must be 1D with values >= 0")
    return N, T


def _resolve_cores(cores) -> int:
    cpu = os.cpu_count() or 1
    if cores is None:
        return cpu
    try:
        cores = int(cores)
    except Exception:
        return cpu
    if cores <= 0:
        return cpu
    return min(cores, cpu)


def _sample_smc(
    *,
    draws: int,
    chains: int,
    cores=None,
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
):
    trace = pm.sample_smc(
        draws=int(draws),
        chains=int(chains),
        cores=_resolve_cores(cores),
        random_seed=random_seed,
        progressbar=True,
        return_inferencedata=False,
        threshold=float(threshold),
        correlation_threshold=float(correlation_threshold),
    )

    to_idata = getattr(pm, "to_inferencedata", None) or getattr(pm, "to_inference_data", None)
    if to_idata is None:
        raise RuntimeError("PyMC does not expose to_inferencedata/to_inference_data")

    idata = to_idata(trace, log_likelihood=True)
    _store_smc_log_marginal_likelihood(idata, trace, chains=int(chains))
    return idata


def _store_smc_log_marginal_likelihood(idata, trace, *, chains: int) -> None:
    report = getattr(trace, "report", None)
    if report is None or not hasattr(report, "log_marginal_likelihood"):
        return

    logml = _parse_log_marginal_likelihood(report.log_marginal_likelihood, chains=chains)
    if logml is None:
        return

    try:
        idata.sample_stats["log_marginal_likelihood"] = (("chain",), logml)
    except Exception:
        idata.attrs["log_marginal_likelihood"] = logml.tolist()


def _parse_log_marginal_likelihood(raw, *, chains: int) -> Optional[np.ndarray]:
    if not isinstance(raw, (list, tuple, np.ndarray)):
        try:
            return np.full(chains, float(raw), dtype=float)
        except Exception:
            return None

    if all(not isinstance(item, (list, tuple, np.ndarray)) for item in raw):
        arr = np.asarray(raw, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return np.full(chains, float(arr[-1]), dtype=float)
        return None

    vals: list[float] = []
    for item in raw:
        if isinstance(item, (list, tuple, np.ndarray)):
            arr = np.asarray(item, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals.append(float(arr[-1]))
            continue
        try:
            vals.append(float(item))
        except Exception:
            continue

    vals = [val for val in vals if np.isfinite(val)]
    if not vals:
        return None

    logml = np.asarray(vals, dtype=float)
    if logml.size < chains:
        logml = np.concatenate([logml, np.full(chains - logml.size, np.nan)])
    return logml[:chains]


def smc_log_evidence(idata) -> float:
    ss = getattr(idata, "sample_stats", None)
    if ss is not None and "log_marginal_likelihood" in getattr(ss, "data_vars", {}):
        vals = np.asarray(ss["log_marginal_likelihood"].values, dtype=float)
    else:
        attrs = getattr(idata, "attrs", {})
        if "log_marginal_likelihood" not in attrs:
            raise RuntimeError("SMC evidence missing: sample_stats['log_marginal_likelihood'].")
        vals = np.asarray(attrs["log_marginal_likelihood"], dtype=float)

    vals = vals[np.isfinite(vals)]
    if not vals.size:
        raise RuntimeError("Could not parse SMC log evidence values.")
    return float(np.mean(vals))


def _summarize(idata) -> None:
    print(az.summary(idata, hdi_prob=0.95))


def inference_homo(
    contacts_per_cell,
    obs_time: float,
    *,
    draws: int = 3000,
    chains: int = 4,
    cores=None,
    lambda_prior_bounds=(-5.0, 2.0),
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
):
    N, T = _validate_counts(contacts_per_cell, obs_time)

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
        lam = pm.Deterministic("lambda", 10.0**eta)
        pm.Poisson("contacts", mu=lam * T, observed=N)

        idata = _sample_smc(
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            threshold=threshold,
            correlation_threshold=correlation_threshold,
        )
        _summarize(idata)

    return {"idata": idata, "model": model}


def inference_Z2P(
    contacts_per_cell,
    obs_time: float,
    *,
    draws: int = 3000,
    chains: int = 4,
    cores=None,
    lambda_prior_bounds=(-5.0, 2.0),
    p_prior_bounds=(1.0, 1.0),
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
):
    N, T = _validate_counts(contacts_per_cell, obs_time)

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
        p0 = pm.Beta("p_zero", alpha=float(p_prior_bounds[0]), beta=float(p_prior_bounds[1]))
        lam = pm.Deterministic("lambda", 10.0**eta)
        pm.ZeroInflatedPoisson("contacts", psi=1.0 - p0, mu=lam * T, observed=N)

        idata = _sample_smc(
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            threshold=threshold,
            correlation_threshold=correlation_threshold,
        )
        _summarize(idata)

    return {"idata": idata, "model": model}



def _gamma_shape_rate_from_mean_sd(mu, sd):
    sd = pt.maximum(sd, 1e-12)
    var = sd**2
    shape = mu**2 / var
    rate = mu / var
    return shape, rate


def _gamma_marginal_logI(n_t, T: float, mu_lam, sig_lam, *, max_count: int):
    shape, rate = _gamma_shape_rate_from_mean_sd(mu_lam, sig_lam)
    max_count = int(max_count)
    if max_count < 0:
        raise ValueError("max_count must be >= 0")

    # Stable for integer counts:
    # gammaln(shape + n) - gammaln(shape) = sum_{k=0}^{n-1} log(shape + k).
    # This avoids large-term cancellation near sigma_lambda = 0.
    log_rising = pt.zeros_like(n_t)
    for k in range(max_count):
        log_rising = pt.switch(n_t > float(k), log_rising + pt.log(shape + float(k)), log_rising)

    return (
        log_rising
        - n_t * pt.log(rate)
        - (shape + n_t) * pt.log1p(float(T) / rate)
    )
    


def _poisson_count_constant(n_t, T: float):
    return n_t * pt.log(float(T)) - pt.gammaln(n_t + 1.0)

def inference_Dis2P(
    contacts_per_cell,
    obs_time: float,
    *,
    draws: int = 3000,
    chains: int = 4,
    cores=None,
    lambda_prior_bounds=(-5.0, 2.0),
    std_prior_factor: float = 1.0,
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
):
    N, T = _validate_counts(contacts_per_cell, obs_time)
    n_t = pt.as_tensor_variable(N.astype(float))
    max_count = int(N.max(initial=0))

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
        mu_lam = pm.Deterministic("mu_lambda", 10.0**eta)
        sig_lam = pm.HalfNormal("sigma_lambda", sigma=float(std_prior_factor))
        pm.Potential(
            "gamma_marg_counts_ll",
            pt.sum(
                _poisson_count_constant(n_t, T)
                + _gamma_marginal_logI(n_t, T, mu_lam, sig_lam, max_count=max_count)
            ),
        )

        idata = _sample_smc(
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            threshold=threshold,
            correlation_threshold=correlation_threshold,
        )
        _summarize(idata)

    return {"idata": idata, "model": model}


def inference_hetero3(
    contacts_per_cell,
    obs_time: float,
    *,
    draws: int = 3000,
    chains: int = 4,
    cores=None,
    lambda_prior_bounds=(-5.0, 2.0),
    p_prior_bounds=(1.0, 1.0),
    std_prior_factor: float = 1.0,
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
):
    N, T = _validate_counts(contacts_per_cell, obs_time)
    n_t = pt.as_tensor_variable(N.astype(float))
    max_count = int(N.max(initial=0))

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
        mu_lam = pm.Deterministic("mu_lambda", 10.0**eta)
        sig_lam = pm.HalfNormal("sigma_lambda", sigma=float(std_prior_factor))
        p_zero = pm.Beta("p_zero", alpha=float(p_prior_bounds[0]), beta=float(p_prior_bounds[1]))

        log_active = (
            pt.log1p(-p_zero)
            + _poisson_count_constant(n_t, T)
            + _gamma_marginal_logI(n_t, T, mu_lam, sig_lam, max_count=max_count)
        )
        log_zero = pt.logaddexp(
            pt.log(p_zero),
            pt.log1p(-p_zero)
            + _gamma_marginal_logI(pt.zeros_like(n_t), T, mu_lam, sig_lam, max_count=0),
        )
        is_zero = pt.as_tensor_variable((N == 0).astype(bool))
        pm.Potential("zi_gamma_marg_counts_ll", pt.sum(pt.switch(is_zero, log_zero, log_active)))

        idata = _sample_smc(
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            threshold=threshold,
            correlation_threshold=correlation_threshold,
        )
        _summarize(idata)

    return {"idata": idata, "model": model}
