import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
import diffrax
import optax
import numpy as np


def sirc_vector_field(t, y, args):
    """SIRC model vector field for Diffrax.
    compartments: S, I, R, C + CumI (cumulative incidence tracker).
    CumI doesn't participate in the dynamics — it just accumulates new infections
    so we can compute attack rate correctly even when delta feeds C back into S"""
    S, I, R, C, CumI = y
    beta, gamma, sigma, delta, mu = args
    N = S + I + R + C          # CumI is a counter, not a population compartment

    new_inf = beta * S * I / N

    dS = -new_inf + delta * C + mu * (N - S)
    dI =  new_inf - gamma * I - mu * I
    dR =  gamma * I - sigma * R - mu * R
    dC =  sigma * R - delta * C - mu * C
    dCumI = new_inf             # accumulate true incidence

    return jnp.array([dS, dI, dR, dC, dCumI])


def sirc_vector_field_beta_forced(t, y, args):
    """SIRC vector field with a time-varying beta path.

    we use this for weekly modelling when drift is only available monthly/weekly.
    in that case drift cannot sensibly keep changing S0 after the season starts,
    so we let it modulate beta(t) instead."""
    beta_ts, beta_vals, gamma, sigma, delta, mu = args
    beta_t = jnp.interp(t, beta_ts, beta_vals)
    return sirc_vector_field(t, y, (beta_t, gamma, sigma, delta, mu))


@jit
def simulate_season(beta, gamma, sigma, delta, mu, S0, ihr=0.015,
                    duration=180.0, n_save=500):
    """simulate one flu season with Diffrax (JIT-compiled).
    returns (peak_I, attack_rate, hosp_rate, sol)"""
    I0 = 1e-4
    # start pre-season protection in R rather than C so the model does not
    # immediately rely on slow C->S recycling to generate susceptibility.
    R0 = jnp.maximum(1.0 - S0 - I0, 0.0)
    C0 = 0.0
    y0 = jnp.array([S0, I0, R0, C0, 0.0])   # 5th element = CumI starts at 0

    term = diffrax.ODETerm(sirc_vector_field)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, duration, n_save))

    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=duration, dt0=0.25, y0=y0,
        args=(beta, gamma, sigma, delta, mu),
        stepsize_controller=controller,
        saveat=saveat,
        max_steps=65536,
    )

    peak_I = jnp.max(sol.ys[:, 1])
    # use cumulative incidence (5th compartment) — the old S0-S(end) formula
    # gave zero whenever C->S inflow exceeded the epidemic's S->I outflow
    attack_rate = sol.ys[-1, 4]
    hosp_rate = attack_rate * ihr * 100_000

    return peak_I, attack_rate, hosp_rate, sol


def drift_to_delta(drift, delta_min, delta_max):
    """Map normalized drift score [0,1] -> delta (C->S rate).
    kept for backwards compatibility with older notebook checkpoints that still
    parameterise drift through delta."""
    return delta_min + (delta_max - delta_min) * drift


def drift_to_s0(drift, s0_min, s0_max):
    """Map normalized drift score [0,1] -> season-specific S0.

    we now let drift change the effective susceptible pool at season start,
    because that is the cleaner immune-escape interpretation for a single-strain
    seasonal model than speeding up within-season C -> S recycling."""
    return s0_min + (s0_max - s0_min) * drift


def drift_to_beta(drift, beta_min, beta_max, ve=0.0):
    """Map normalized drift score [0,1] -> season-specific beta.

    this is useful when we want drift to change transmission potential rather
    than susceptibility at season start."""
    beta = beta_min + (beta_max - beta_min) * drift
    return beta * (1.0 - ve)


def drift_to_onset_week(drift, onset_week_min, onset_week_max):
    """Map normalized drift to an onset week.

    higher drift means earlier takeoff, so novelty-heavy seasons get smaller
    onset weeks and low-drift seasons get later onsets."""
    return onset_week_max - (onset_week_max - onset_week_min) * drift


def drift_path_to_beta_weekly(drift_path, ve, beta_drift_scale):
    """Map a within-season drift path to a beta path for weekly simulation."""
    beta_base = BETA_FIXED * (1.0 - ve)
    beta_weekly = beta_base * (1.0 + beta_drift_scale * drift_path)
    return jnp.maximum(beta_weekly, 1e-6)


def apply_onset_to_beta_weekly(beta_weekly, onset_week, sharpness=1.5):
    """Apply a smooth onset gate to a weekly beta path.

    using a sigmoid keeps the onset differentiable so weekly calibration can
    learn timing rather than being stuck with a hard discrete week shift."""
    week_idx = jnp.arange(1, beta_weekly.shape[0] + 1, dtype=beta_weekly.dtype)
    gate = jax.nn.sigmoid(sharpness * (week_idx - onset_week))
    return jnp.maximum(beta_weekly * gate, 1e-6)


# note: in the SIRC model, the effective R0 is not simply beta/gamma due to the
# C->S recycling pathway — Casagrandi (2006) shows R0 depends on the endemic
# equilibrium. our BETA_FIXED/GAMMA_FIXED = 1.5 is an approximation for the
# single-season regime where C->S recycling is small.
BETA_FIXED = 0.30         # R0 ~ 1.5 with gamma=0.2 — a defensible seasonal-flu baseline
GAMMA_FIXED = 1.0 / 5.0      # 5-day mean infectious period (Biggerstaff et al. 2014)
# mu contributes ~0.3% population turnover over a 180-day season — negligible for
# single-season dynamics but included for consistency with Casagrandi (2006).
MU_FIXED = 1.0 / (75 * 365)  # demographic turnover, ~75yr life expectancy
# sigma fixed from literature instead of fitted — Casagrandi (2006) uses ~120 days
# as baseline. fitting sigma alongside delta created identifiability problems
# (both control how fast individuals return to S; only the C sojourn differs).
SIGMA_FIXED = 1.0 / 120.0    # R->C rate, ~120-day cross-immunity development
# we keep delta fixed in the new setup because drift is doing immune-escape work
# through S0 now, so letting delta float as well just confounds two immunity knobs
DELTA_FIXED = 1.0 / 365.0    # slow C->S recycling over roughly a year


def _resolve_s0_and_delta(params, drift):
    """Resolve season-specific S0 and delta from a params dict.

    we support both the new parameterisation (drift -> S0, fixed delta) and the
    old one (drift -> delta, shared S0_base) so older notebook cells still run."""
    if 'S0_min' in params and 'S0_max' in params:
        s0 = drift_to_s0(drift, params['S0_min'], params['S0_max'])
        delta = params.get('delta_fixed', DELTA_FIXED)
        return s0, delta

    if 'S0_base' in params and 'delta_fixed' in params:
        return params['S0_base'], params['delta_fixed']

    # older checkpoints still pass S0_base + delta_min/max, so we keep that path
    # alive rather than breaking the notebook outright
    s0 = params['S0_base']
    delta = drift_to_delta(drift, params['delta_min'], params['delta_max'])
    return s0, delta


def _resolve_beta(params, drift, ve):
    """Resolve season-specific beta from params if drift is mapped to beta."""
    if 'beta_min' in params and 'beta_max' in params:
        return drift_to_beta(drift, params['beta_min'], params['beta_max'], ve=ve)
    return BETA_FIXED * (1.0 - ve)


def _resolve_onset_week(params, drift):
    """Resolve a season-specific onset week when weekly timing is being fitted."""
    if 'onset_week_min' in params and 'onset_week_max' in params:
        return drift_to_onset_week(drift, params['onset_week_min'], params['onset_week_max'])
    return None


@jit
def simulate_batch(params, drift_vec, ve_vec=None):
    """simulate SIRC for all seasons in parallel via vmap.
    ve_vec is a rough proxy — true VE varies 10-60% across seasons
    and we only have coverage data, not effectiveness estimates"""
    ihr = params['ihr']

    # if no VE data provided, use zeros (no adjustment)
    if ve_vec is None:
        ve_vec = jnp.zeros_like(drift_vec)

    def _sim_one(drift, ve):
        s0, delta = _resolve_s0_and_delta(params, drift)
        beta_eff = _resolve_beta(params, drift, ve)
        _, _, hosp, _ = simulate_season(
            beta_eff, GAMMA_FIXED, SIGMA_FIXED, delta, MU_FIXED, s0, ihr=ihr
        )
        return hosp

    return vmap(_sim_one)(drift_vec, ve_vec)


# reduced from 5 to 4 free params by fixing sigma_inv and replacing k_map with ihr.
# with ~8 calibration seasons this gives a 4:8 = 1:2 param-to-data ratio,
# which is tight but defensible (Raue et al. 2009 recommend >=3:1 data:param).
DEFAULT_BOUNDS = {
    'S0_min':      (0.20, 0.55),
    'S0_max':      (0.45, 0.90),
    'delta_fixed': (DELTA_FIXED, DELTA_FIXED),
    'ihr':         (0.008, 0.025),   # CDC burden estimates: IHR ranges 0.8-2.5% across seasons
}


BETA_DRIFT_BOUNDS = {
    # keep the weekly/seasonal beta search in a regime that can actually grow
    # from realistic S0 values under gamma = 1/5.
    'beta_min':    (0.34, 0.50),
    'beta_max':    (0.50, 0.90),
    'S0_base':     (0.60, 0.85),
    'onset_week_min': (1.0, 8.0),
    'onset_week_max': (8.0, 18.0),
    'delta_fixed': (DELTA_FIXED, DELTA_FIXED),
    'ihr':         (0.008, 0.025),
}


def constrain_params(raw, bounds=None):
    """map unconstrained raw params -> valid ranges via sigmoid"""
    if bounds is None:
        bounds = DEFAULT_BOUNDS
    out = {}
    for k, (lo, hi) in bounds.items():
        if hi == lo:
            out[k] = jnp.asarray(lo, dtype=jnp.float32)
        else:
            out[k] = jax.nn.sigmoid(raw[k]) * (hi - lo) + lo
    return out


def init_raw_params(bounds=None):
    """initialise unconstrained parameters at midpoints (sigmoid(0) = 0.5)"""
    if bounds is None:
        bounds = DEFAULT_BOUNDS
    return {k: jnp.float32(0.0) for k in bounds}


def calibrate(calib_drift, calib_hosp_obs, n_steps=500, lr=0.05, bounds=None,
              raw_init=None, ve_vec=None):
    """calibrate SIRC parameters via Adam + autodiff through the ODE solver.
    we use RRMSE instead of absolute RMSE so mild seasons (~50/100k) and
    severe seasons (~200/100k) contribute proportionally"""
    raw_params = init_raw_params(bounds) if raw_init is None else raw_init

    @jit
    def loss_fn(raw):
        params = constrain_params(raw, bounds)
        hosp_sim = simulate_batch(params, calib_drift, ve_vec)
        # relative RMSE — gives proportional weight to mild and severe seasons
        return jnp.sqrt(jnp.mean(((hosp_sim - calib_hosp_obs) / calib_hosp_obs) ** 2))

    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_steps)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(raw_params)

    loss_history = []
    for step in range(n_steps):
        loss, grads = value_and_grad(loss_fn)(raw_params)
        updates, opt_state = optimizer.update(grads, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)
        loss_history.append(float(loss))
        if (step + 1) % 100 == 0:
            p = constrain_params(raw_params, bounds)
            if 'beta_min' in p and 'beta_max' in p:
                print(f'  Step {step+1:4d}: RRMSE={float(loss):.4f}, '
                      f'beta_range=({float(p["beta_min"]):.3f}, {float(p["beta_max"]):.3f}), '
                      f'S0_base={float(p["S0_base"]):.3f}, IHR={float(p["ihr"]):.4f}')
            else:
                print(f'  Step {step+1:4d}: RRMSE={float(loss):.4f}, '
                      f'beta={BETA_FIXED:.3f} (fixed), '
                      f'S0_range=({float(p["S0_min"]):.3f}, {float(p["S0_max"]):.3f}), '
                      f'IHR={float(p["ihr"]):.4f}')

    opt_params = constrain_params(raw_params, bounds)
    return opt_params, raw_params, loss_history


def calibrate_multistart(calib_drift, calib_hosp_obs, n_starts=5,
                         n_steps=500, lr=0.05, bounds=None, seed=42,
                         ve_vec=None):
    """run calibration from multiple random starts and keep the best fit.
    with a tiny number of seasons the loss surface can be flat, so midpoint-only
    initialisation is riskier than it looks."""
    if bounds is None:
        bounds = DEFAULT_BOUNDS

    rng = np.random.default_rng(seed)
    best_result = None
    best_loss = float('inf')

    for i in range(n_starts):
        raw_init = {
            k: jnp.float32(rng.uniform(-2.0, 2.0))
            for k in bounds
        }
        print(f'== Multistart {i + 1}/{n_starts} ==')
        opt_params, raw_params, loss_history = calibrate(
            calib_drift,
            calib_hosp_obs,
            n_steps=n_steps,
            lr=lr,
            bounds=bounds,
            raw_init=raw_init,
            ve_vec=ve_vec,
        )
        final_loss = loss_history[-1]
        if final_loss < best_loss:
            best_loss = final_loss
            best_result = (opt_params, raw_params, loss_history)

    return best_result


def calibrate_loo_cv(drift_vec, hosp_obs_vec, n_steps=500, lr=0.05,
                     bounds=None, n_starts=3, seed=42, ve_vec=None,
                     season_labels=None):
    """leave-one-season-out CV — each season is held out in turn and predicted
    from a model fit on the remaining seasons. reports per-fold and mean RRMSE.
    standard in small-sample epidemic forecasting (Biggerstaff et al. 2016)."""
    if bounds is None:
        bounds = DEFAULT_BOUNDS

    n = len(drift_vec)
    results = []

    for i in range(n):
        # hold out season i
        mask = jnp.arange(n) != i
        train_drift = drift_vec[mask]
        train_hosp = hosp_obs_vec[mask]
        train_ve = ve_vec[mask] if ve_vec is not None else None

        opt_params, _, _ = calibrate_multistart(
            train_drift, train_hosp, n_starts=n_starts,
            n_steps=n_steps, lr=lr, bounds=bounds, seed=seed + i,
            ve_vec=train_ve,
        )

        # predict held-out season
        held_drift = drift_vec[i:i+1]
        held_ve = ve_vec[i:i+1] if ve_vec is not None else None
        hosp_pred = float(simulate_batch(opt_params, held_drift, held_ve)[0])
        hosp_true = float(hosp_obs_vec[i])
        rel_err = abs(hosp_pred - hosp_true) / hosp_true

        label = season_labels[i] if season_labels is not None else f'fold_{i}'
        results.append({
            'season': label,
            'hosp_obs': hosp_true,
            'hosp_pred': hosp_pred,
            'abs_err': abs(hosp_pred - hosp_true),
            'rel_err': rel_err,
        })
        print(f'  LOO fold {i+1}/{n} ({label}): '
              f'obs={hosp_true:.1f}, pred={hosp_pred:.1f}, '
              f'rel_err={rel_err:.3f}')

    # summary
    rel_errs = [r['rel_err'] for r in results]
    abs_errs = [r['abs_err'] for r in results]
    print(f'\nLOO-CV summary:')
    print(f'  Mean relative error: {np.mean(rel_errs):.3f} +/- {np.std(rel_errs):.3f}')
    print(f'  Mean absolute error: {np.mean(abs_errs):.1f} +/- {np.std(abs_errs):.1f} /100k')

    return results


N_WEEKS_PER_SEASON = 30  # max MMWR weeks in a standard flu season (wk 40 → wk 20)


@jit
def simulate_season_weekly(beta, gamma, sigma, delta, mu, S0, ihr=0.015):
    """simulate one flu season and return weekly hospitalisation incidence /100k.
    calibrating against this curve gives ~N_WEEKS_PER_SEASON x more signal per
    season than the seasonal-cumulative target, and forces the model to
    reproduce peak timing as well as season severity"""
    I0 = 1e-4
    R0 = jnp.maximum(1.0 - S0 - I0, 0.0)
    C0 = 0.0
    y0 = jnp.array([S0, I0, R0, C0, 0.0])

    # save CumI at the boundary of each 7-day window
    week_ts = jnp.arange(0, N_WEEKS_PER_SEASON + 1, dtype=jnp.float32) * 7.0

    term       = diffrax.ODETerm(sirc_vector_field)
    solver     = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)
    saveat     = diffrax.SaveAt(ts=week_ts)

    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=float(N_WEEKS_PER_SEASON * 7),
        dt0=0.25, y0=y0, args=(beta, gamma, sigma, delta, mu),
        stepsize_controller=controller, saveat=saveat, max_steps=100_000,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )

    # CumI = compartment 4; weekly new infections = Δ(CumI) over each 7-day window
    cum_i      = sol.ys[:, 4]          # (N_WEEKS_PER_SEASON + 1,)
    weekly_inc = jnp.diff(cum_i)       # (N_WEEKS_PER_SEASON,)
    return weekly_inc * ihr * 100_000  # hospitalisations per 100k per week


@jit
def simulate_season_weekly_forced(beta_weekly, gamma, sigma, delta, mu, S0, ihr=0.015):
    """Simulate one season with a time-varying weekly beta path."""
    I0 = 1e-4
    R0 = jnp.maximum(1.0 - S0 - I0, 0.0)
    C0 = 0.0
    y0 = jnp.array([S0, I0, R0, C0, 0.0])

    week_ts = jnp.arange(0, N_WEEKS_PER_SEASON + 1, dtype=jnp.float32) * 7.0
    # we repeat the last weekly value at the final boundary so the interpolation
    # has the same number of support points as week_ts
    beta_vals = jnp.concatenate([beta_weekly, beta_weekly[-1:]])

    term       = diffrax.ODETerm(sirc_vector_field_beta_forced)
    solver     = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)
    saveat     = diffrax.SaveAt(ts=week_ts)

    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=float(N_WEEKS_PER_SEASON * 7),
        dt0=0.25, y0=y0,
        args=(week_ts, beta_vals, gamma, sigma, delta, mu),
        stepsize_controller=controller, saveat=saveat, max_steps=65536,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )

    cum_i = sol.ys[:, 4]
    weekly_inc = jnp.diff(cum_i)
    return weekly_inc * ihr * 100_000


@jit
def simulate_batch_weekly(params, drift_vec, ve_vec=None):
    """simulate SIRC for all seasons in parallel, returning weekly hosp rates.
    drift_vec can be [n_seasons] for season-level or [n_seasons, N_WEEKS_PER_SEASON]
    for within-season drift paths"""
    ihr       = params['ihr']

    if ve_vec is None:
        ve_vec = jnp.zeros((drift_vec.shape[0],), dtype=drift_vec.dtype) if drift_vec.ndim == 2 else jnp.zeros_like(drift_vec)

    if drift_vec.ndim == 1:
        def _sim_one(drift, ve):
            s0, delta = _resolve_s0_and_delta(params, drift)
            beta_eff = _resolve_beta(params, drift, ve)
            onset_week = _resolve_onset_week(params, drift)
            if onset_week is None:
                return simulate_season_weekly(
                    beta_eff, GAMMA_FIXED, SIGMA_FIXED, delta, MU_FIXED, s0, ihr=ihr,
                )
            beta_weekly = apply_onset_to_beta_weekly(
                jnp.full((N_WEEKS_PER_SEASON,), beta_eff, dtype=jnp.float32),
                onset_week,
            )
            return simulate_season_weekly_forced(
                beta_weekly, GAMMA_FIXED, SIGMA_FIXED, delta, MU_FIXED, s0, ihr=ihr,
            )

        return vmap(_sim_one)(drift_vec, ve_vec)

    beta_drift_scale = params['beta_drift_scale'] if 'beta_drift_scale' in params else jnp.float32(0.0)

    # with a drift path we still need one season-level susceptibility anchor,
    # so we use the mean novelty across the season for S0 and let beta(t) carry
    # the within-season timing information
    def _sim_one_path(drift_path, ve):
        season_drift = jnp.mean(drift_path)
        s0, delta = _resolve_s0_and_delta(params, season_drift)
        beta_weekly = drift_path_to_beta_weekly(drift_path, ve, beta_drift_scale)
        return simulate_season_weekly_forced(
            beta_weekly, GAMMA_FIXED, SIGMA_FIXED, delta, MU_FIXED, s0, ihr=ihr,
        )

    return vmap(_sim_one_path)(drift_vec, ve_vec)

WEEKLY_BOUNDS = {                                                                                                                                      
      'S0_min':      (0.45, 0.75),
      'S0_max':      (0.70, 0.98),    # we keep the upper end high so novelty-heavy seasons can still take off
      'delta_fixed': (DELTA_FIXED, DELTA_FIXED),
      'beta_drift_scale': (0.0, 1.5),  # lets monthly drift move the epidemic within a season
      'ihr':         (0.008, 0.025),                                                                                                                     
  }   

WEEKLY_BETA_DRIFT_BOUNDS = {
      'beta_min':    (0.34, 0.50),
      'beta_max':    (0.50, 0.90),
      'S0_base':     (0.60, 0.85),
      'onset_week_min': (1.0, 8.0),
      'onset_week_max': (8.0, 18.0),
      'delta_fixed': (DELTA_FIXED, DELTA_FIXED),
      'ihr':         (0.008, 0.025),
}

def calibrate_weekly(calib_drift, obs_weekly, obs_mask, n_steps=500, lr=0.05,
                     bounds=None, raw_init=None, ve_vec=None):
    """calibrate SIRC parameters against weekly hospitalisation time series.
    RRMSE is computed only over weeks where FluSurv-NET has data (obs_mask == True),
    giving ~N_WEEKS_PER_SEASON x more signal than seasonal-cumulative calibration"""
    if bounds is None:
        bounds = WEEKLY_BOUNDS

    raw_params = init_raw_params(bounds) if raw_init is None else raw_init

    if ve_vec is None:
        ve_vec = jnp.zeros_like(calib_drift)

    @jit
    def loss_fn(raw):
        params   = constrain_params(raw, bounds)
        hosp_sim = simulate_batch_weekly(params, calib_drift, ve_vec)
        # the old loss treated every positive week almost equally, so missing the
        # peak could cost about the same as missing a tiny shoulder week. that is
        # why we kept getting those flat-ish solutions that looked "fine" to Adam
        # but awful to us on the plots :(
        active = obs_mask
        peak = jnp.max(jnp.where(obs_mask, obs_weekly, 0.0), axis=1, keepdims=True)
        safe_floor = 0.05 * peak + 0.1
        safe_denom = jnp.maximum(obs_weekly, safe_floor)

        # we upweight high-incidence weeks using the observed curve itself.
        # this keeps the loss relative, but makes the epidemic peak matter more
        # than a 0.1/100k week in the tail
        peak_scale = jnp.maximum(peak, 1e-6)
        peak_weight = 0.20 + 0.80 * (obs_weekly / peak_scale) ** 2

        rel_sq = jnp.where(active, ((hosp_sim - obs_weekly) / safe_denom) ** 2, 0.0)
        weighted_rel_sq = jnp.where(active, peak_weight * rel_sq, 0.0)
        weight_total = jnp.maximum(jnp.where(active, peak_weight, 0.0).sum(), 1.0)
        loss = jnp.sqrt(weighted_rel_sq.sum() / weight_total)
        # NaN guard: if anything goes numerically bad, return a large finite loss so Adam recovers
        loss = jnp.where(jnp.isfinite(loss), loss, 1e6)
        return loss

    schedule  = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_steps)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(raw_params)

    loss_history = []
    for step in range(n_steps):
        loss, grads = value_and_grad(loss_fn)(raw_params)
        updates, opt_state = optimizer.update(grads, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)
        loss_history.append(float(loss))
        if (step + 1) % 100 == 0:
            p = constrain_params(raw_params, bounds)
            if 'beta_min' in p and 'beta_max' in p:
                print(f'  Step {step+1:4d}: weekly RRMSE={float(loss):.4f}, '
                      f'beta_range=({float(p["beta_min"]):.3f}, {float(p["beta_max"]):.3f}), '
                      f'S0_base={float(p["S0_base"]):.3f}, '
                      f'onset_range=({float(p["onset_week_min"]):.2f}, {float(p["onset_week_max"]):.2f}), '
                      f'IHR={float(p["ihr"]):.4f}')
            else:
                print(f'  Step {step+1:4d}: weekly RRMSE={float(loss):.4f}, '
                      f'S0_range=({float(p["S0_min"]):.3f}, {float(p["S0_max"]):.3f}), '
                      f'beta_drift_scale={float(p["beta_drift_scale"]):.3f}, '
                      f'IHR={float(p["ihr"]):.4f}')

    opt_params = constrain_params(raw_params, bounds)
    return opt_params, raw_params, loss_history


def calibrate_weekly_multistart(calib_drift, obs_weekly, obs_mask, n_starts=5,
                                n_steps=500, lr=0.05, bounds=None, seed=42,
                                ve_vec=None):
    """Multi-start wrapper for calibrate_weekly — same interface as calibrate_multistart."""
    if bounds is None:
        bounds = WEEKLY_BOUNDS

    rng = np.random.default_rng(seed)
    best_result = None
    best_loss   = float('inf')

    for i in range(n_starts):
        raw_init = {k: jnp.float32(rng.uniform(-2.0, 2.0)) for k in bounds}
        print(f'== Multistart {i + 1}/{n_starts} ==')
        result = calibrate_weekly(
            calib_drift, obs_weekly, obs_mask,
            n_steps=n_steps, lr=lr, bounds=bounds,
            raw_init=raw_init, ve_vec=ve_vec,
        )
        if result[2][-1] < best_loss:
            best_loss   = result[2][-1]
            best_result = result

    return best_result
