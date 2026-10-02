import numpy as np
import pandas as pd
from tqdm import tqdm


def poisson_disk_samples_rectangle(width, height, r, k=30, seed=None, max_points=None):
    """Uniformly distribute as many non-overlapping points as possible (min dist r) in rectangle."""
    rng = np.random.default_rng(seed)
    cell_size = r / np.sqrt(2)
    grid_shape = (int(np.ceil(width/cell_size)), int(np.ceil(height/cell_size)))
    grid = -np.ones(grid_shape, dtype=int)
    samples, active = [], []
    pt = rng.uniform([0,0], [width,height])
    samples.append(pt)
    grid_idx = (pt // cell_size).astype(int)
    grid[tuple(grid_idx)] = 0
    active.append(0)
    while active and (max_points is None or len(samples) < max_points):
        idx = rng.choice(active)
        centre = samples[idx]
        found = False
        for _ in range(k):
            angle = rng.uniform(0, 2*np.pi)
            rad   = rng.uniform(r, 2*r)
            offset = rad * np.array([np.cos(angle), np.sin(angle)])
            cand = centre + offset
            if not (0 <= cand[0] < width and 0 <= cand[1] < height): continue
            grid_x, grid_y = (cand // cell_size).astype(int)
            x0, x1 = max(0, grid_x-2), min(grid_shape[0], grid_x+3)
            y0, y1 = max(0, grid_y-2), min(grid_shape[1], grid_y+3)
            ok = True
            for ix in range(x0, x1):
                for iy in range(y0, y1):
                    sidx = grid[ix, iy]
                    if sidx != -1:
                        dist2 = np.sum((samples[sidx] - cand)**2)
                        if dist2 < r**2:
                            ok = False
                            break
                if not ok: break
            if ok:
                samples.append(cand)
                grid[grid_x, grid_y] = len(samples)-1
                active.append(len(samples)-1)
                found = True
                break
        if not found:
            active.remove(idx)
    return np.array(samples)

def poisson_disk_samples_circle(centre, radius, r, k=30, seed=None, max_points=None):
    """Uniformly distribute as many non-overlapping points as possible (min dist r) in a circle."""
    rng = np.random.default_rng(seed)
    samples, active = [], []
    theta = rng.uniform(0, 2*np.pi)
    rad   = rng.uniform(0, radius)
    pt = np.array(centre) + rad * np.array([np.cos(theta), np.sin(theta)])
    samples.append(pt)
    active.append(0)
    while active and (max_points is None or len(samples) < max_points):
        idx = rng.choice(active)
        centre_pt = samples[idx]
        found = False
        for _ in range(k):
            angle = rng.uniform(0, 2*np.pi)
            rad   = rng.uniform(r, 2*r)
            cand = centre_pt + rad * np.array([np.cos(angle), np.sin(angle)])
            if np.linalg.norm(cand - centre) > radius: continue
            ok = True
            for s in samples:
                if np.linalg.norm(s-cand) < r:
                    ok = False
                    break
            if ok:
                samples.append(cand)
                active.append(len(samples)-1)
                found = True
                break
        if not found:
            active.remove(idx)
    return np.array(samples)

def gaussian_samples_rectangle(width, height, N, sigma, r, seed=None, max_trials=100000):
    """Sample N points from 2D normal, clipped to rectangle, min dist r."""
    rng = np.random.default_rng(seed)
    cx, cy = width/2, height/2
    points = []
    trials = 0
    while len(points) < N and trials < max_trials:
        pt = rng.normal([cx, cy], sigma, 2)
        if not (0 <= pt[0] < width and 0 <= pt[1] < height):
            trials += 1; continue
        if all(np.linalg.norm(pt - np.array(p)) >= r for p in points):
            points.append(pt)
        trials += 1
    if len(points) < N:
        raise RuntimeError(f"Could only place {len(points)} non-overlapping points out of {N} requested. Try fewer cells or smaller radii.")
    return np.array(points)

def gaussian_samples_circle(centre, region_radius, N, sigma, r, seed=None, max_trials=100000):
    """Sample N points from 2D normal in circle, min dist r."""
    rng = np.random.default_rng(seed)
    points = []
    trials = 0
    while len(points) < N and trials < max_trials:
        pt = rng.normal(centre, sigma, 2)
        if np.linalg.norm(pt - centre) > region_radius:
            trials += 1; continue
        if all(np.linalg.norm(pt - np.array(p)) >= r for p in points):
            points.append(pt)
        trials += 1
    if len(points) < N:
        raise RuntimeError(f"Could only place {len(points)} non-overlapping points out of {N} requested. Try fewer cells or smaller radii or higher sigma.")
    return np.array(points)

# --- Main function ---
def generate_two_populations(
    N1: int, N2: int,
    radius1: float, radius2: float,
    shape: str,
    mode: str = "uniform",  # 'uniform' (poisson-disk) or 'gaussian'
    sigma: float = None,    # only used for gaussian mode
    *,
    bounds=None, centre=None, region_radius=None,
    seed=None,
    min_distance_factor: float = 1.0
):
    """Generate two non-overlapping populations, uniform or gaussian."""
    min_sep_KK = 2 * radius1 * min_distance_factor
    min_sep_TT = 2 * radius2 * min_distance_factor
    min_sep_KT = (radius1 + radius2) * min_distance_factor
    min_sep = max(min_sep_KK, min_sep_TT, min_sep_KT)
    # min_sep = max(2*radius1, 2*radius2, radius1+radius2)
    total_cells = N1 + N2

    # --- Rectangle/Square ---
    if shape.lower() in ("rectangle","square"):
        if bounds is None:
            raise ValueError("bounds must be provided")
        x_min, x_max, y_min, y_max = bounds
        width = x_max - x_min
        height = y_max - y_min
        if mode == "uniform":
            pts = poisson_disk_samples_rectangle(width, height, min_sep, seed=seed)
            pts[:,0] += x_min; pts[:,1] += y_min
        elif mode == "gaussian":
            if sigma is None:
                sigma = min(width, height) / 6
            pts = gaussian_samples_rectangle(width, height, total_cells, sigma, min_sep, seed=seed)
            pts[:,0] += x_min; pts[:,1] += y_min
        else:
            raise ValueError("Unknown mode")
    # --- Circle ---
    elif shape.lower() == "circle":
        if centre is None or region_radius is None:
            raise ValueError("centre and region_radius must be provided")
        if mode == "uniform":
            pts = poisson_disk_samples_circle(centre, region_radius, min_sep, seed=seed)
        elif mode == "gaussian":
            if sigma is None:
                sigma = region_radius / 3
            pts = gaussian_samples_circle(centre, region_radius, total_cells, sigma, min_sep, seed=seed)
        else:
            raise ValueError("Unknown mode")
    else:
        raise ValueError("Unsupported shape")

    # --- Max cell number check ---
    if mode == "uniform":
        max_possible = len(pts)
        if total_cells > max_possible:
            raise ValueError(
                f"Too many cells! For these radii and region, max is {max_possible}. "
                f"(You requested {total_cells}.)"
            )
    elif mode == "gaussian":
        if len(pts) < total_cells:
            raise RuntimeError(
                f"Could only place {len(pts)} out of {total_cells}. "
                "Try fewer cells or smaller radii or higher sigma."
            )
    np.random.shuffle(pts)
    return pts[:N1], pts[N1:N1+N2]

def apply_boundary(pos, pol, BOX_X_MIN, BOX_X_MAX, BOX_Y_MIN, BOX_Y_MAX, boundary_condition, cell_radius):
    """
    Confined: mirror-reflect the cell centre at the edge so the cell never crosses the wall.
    The *edge* (not the centre) touches the wall.
    """
    cell_radius = np.broadcast_to(cell_radius, (pos.shape[0],))
    if boundary_condition == 'periodic':
        pos[:, 0] = BOX_X_MIN + (pos[:, 0] - BOX_X_MIN) % (BOX_X_MAX - BOX_X_MIN)
        pos[:, 1] = BOX_Y_MIN + (pos[:, 1] - BOX_Y_MIN) % (BOX_Y_MAX - BOX_Y_MIN)
        # polarity unchanged
    elif boundary_condition == 'confined':
        # Repeat until all are inside
        for _ in range(3):  # usually 1-2 is enough, 3 is bulletproof
            # X walls
            over_left = pos[:, 0] < (BOX_X_MIN + cell_radius)
            if np.any(over_left):
                pos[over_left, 0] = 2*(BOX_X_MIN + cell_radius[over_left]) - pos[over_left, 0]
                pol[over_left, 0] *= -1
            over_right = pos[:, 0] > (BOX_X_MAX - cell_radius)
            if np.any(over_right):
                pos[over_right, 0] = 2*(BOX_X_MAX - cell_radius[over_right]) - pos[over_right, 0]
                pol[over_right, 0] *= -1
            # Y walls
            over_bot = pos[:, 1] < (BOX_Y_MIN + cell_radius)
            if np.any(over_bot):
                pos[over_bot, 1] = 2*(BOX_Y_MIN + cell_radius[over_bot]) - pos[over_bot, 1]
                pol[over_bot, 1] *= -1
            over_top = pos[:, 1] > (BOX_Y_MAX - cell_radius)
            if np.any(over_top):
                pos[over_top, 1] = 2*(BOX_Y_MAX - cell_radius[over_top]) - pos[over_top, 1]
                pol[over_top, 1] *= -1
    return pos, pol

#def lj_force(r_ij, epsilon, sigma):
    #r_norm = np.linalg.norm(r_ij)
    #if r_norm == 0:
        #return np.zeros(2)
    #F_LJ = 24 * epsilon * ((2 * (sigma ** 12) / (r_norm ** 13)) - ((sigma ** 6) / (r_norm ** 7)))
    #return (F_LJ / r_norm) * r_ij

def hook_repulsion_force(r_ij, Krep, L): #L = KILLER_RADIUS + TARGET_RADIUS
    delta = np.linalg.norm(r_ij)

    if delta == 0:
        return np.zeros(2)
    if delta < L:
        F_rep = Krep * (L - delta)
    else:
        F_rep = 0.0

    return (F_rep / delta) * r_ij



def hook_net_force(r_ij, Krep, L, F_A, bound):
    delta = np.linalg.norm(r_ij)

    if delta == 0:
        return np.zeros(2)

    e_ij = r_ij / delta

    # The caller applies Rcap; bound adhesion also acts during overlap.
    force_magnitude = Krep * max(L - delta, 0.0)
    if bound:
        force_magnitude -= F_A

    return force_magnitude * e_ij



def calculate_ij_forces(
    N_KILLER, N_TARGET,
    *,
    killer_positions, target_positions,
    killer_alive, target_alive,
    KREP_KK, KREP_TT, KREP_KT,
    F_A_KK, F_A_TT, F_A_KT,
    KILLER_RADIUS, TARGET_RADIUS,
    is_bound,
    rng,
    k_bind,
    k_unbind,
    dt
):

    N_TOTAL = N_KILLER + N_TARGET

    positions = np.vstack([killer_positions, target_positions])
    labels = np.array([0]*N_KILLER + [1]*N_TARGET)
    alive = np.concatenate([killer_alive, target_alive])

    force = np.zeros((N_TOTAL, N_TOTAL, 2))
    p_bind_step = 1 - np.exp(-k_bind * dt)
    p_unbind_step = 1 - np.exp(-k_unbind * dt)

    for j in range(N_TOTAL):
        if not alive[j]:
            continue

        for k in range(j+1, N_TOTAL):
            if not alive[k]:
                continue

            r_ij = positions[j] - positions[k]
            dist = np.linalg.norm(r_ij)

            # default
            bound = False


            # parameter selection
            if labels[j] == 0 and labels[k] == 0:
                Krep, F_A = KREP_KK, F_A_KK
                L = KILLER_RADIUS * 2
                
            elif labels[j] == 1 and labels[k] == 1:
                Krep, F_A = KREP_TT, F_A_TT
                L = TARGET_RADIUS * 2
                
            else:
                Krep, F_A = KREP_KT, F_A_KT
                L = KILLER_RADIUS + TARGET_RADIUS
                
            
            Rcap = 1.5 * L #from the paper
            contact_tol = 0.05 * L
            bind_possible = dist <= (L + contact_tol)

            # killer-target binding logic
            if labels[j] == 0 and labels[k] == 1:

                killer_idx = j
                target_idx = k - N_KILLER

                if is_bound[killer_idx, target_idx]:

                    if dist > Rcap or rng.random() < p_unbind_step:
                        is_bound[killer_idx, target_idx] = False

                else:

                    if bind_possible and rng.random() < p_bind_step:
                        is_bound[killer_idx, target_idx] = True

                bound = is_bound[killer_idx, target_idx]

            if dist < Rcap:
                f = hook_net_force(r_ij, Krep, L, F_A, bound)
                force[j, k] += f
                force[k, j] -= f

    return force, is_bound



#def calculate_ij_forces(
    #N_KILLER, N_TARGET,
    #*,
    #killer_positions, target_positions, 
    #killer_alive, target_alive,
    #LJ_EPSILON_KK, LJ_EPSILON_TT, LJ_EPSILON_KT,
    #SIGMA_KK, SIGMA_TT, SIGMA_KT):
    #"""
    #Compute the IJ forces, only for alive cells.
    #Returns force array: shape (N_TOTAL, N_TOTAL, 2)
    #"""
    #N_K = N_KILLER
    #N_T = N_TARGET
    #N_TOTAL = N_K + N_T

    #positions = np.vstack([killer_positions, target_positions])
    #labels = np.array([0]*N_K + [1]*N_T)
    #alive = np.concatenate([killer_alive, target_alive])

    #force = np.zeros((N_TOTAL, N_TOTAL, 2))
    #for j in range(N_TOTAL):
        #if not alive[j]:
        #    continue
        #for k in range(j+1, N_TOTAL):
         #   if not alive[k]:
         #       continue
         #   r_ij = positions[j] - positions[k]
         #   # Determine epsilon and sigma
         #   if labels[j] == 0 and labels[k] == 0:
         #       sigma, epsilon = SIGMA_KK, LJ_EPSILON_KK
         #   elif labels[j] == 1 and labels[k] == 1:
         #       sigma, epsilon = SIGMA_TT, LJ_EPSILON_TT
         #  else:
         #       sigma, epsilon = SIGMA_KT, LJ_EPSILON_KT
         #   dist = np.linalg.norm(r_ij)
         #   if 0 < dist <= 2.5 * sigma:
         #       f = lj_force(r_ij, epsilon, sigma)
         #       force[j, k] += f
         #       force[k, j] -= f
    #return force

def KillingProb_ini(N_KILLER, KILL_PROB_INIT=None, KILL_PROBABILISTIC=True):
    """
    Initialise killing probability for all killers.

    Parameters
    ----------
    N_KILLER : int
        Number of killer cells.
    KILL_PROB_INIT : None, float, or (mu, sigma)
        - If None: uniform random in [0.1, 0.9]
        - If scalar: use as constant
        - If (mu, sigma): normal distribution
    KILL_PROBABILISTIC : bool
        - If False: all killers have probability 1.0 (deterministic killing)
        - If True: use the probabilistic initialisation rules above.

    Returns
    -------
    np.ndarray
        Killing probabilities for all killers.
    """
    if not KILL_PROBABILISTIC:
        # No probabilistic decision making — always kill
        return np.ones(N_KILLER, dtype=float)
    else:
        if KILL_PROB_INIT is None:
            return np.random.uniform(0.1, 0.9, size=N_KILLER)
        elif isinstance(KILL_PROB_INIT, (int, float)):
            return np.full(N_KILLER, float(KILL_PROB_INIT))
        elif (
            isinstance(KILL_PROB_INIT, tuple)
            and len(KILL_PROB_INIT) == 2
            and all(isinstance(x, (int, float)) for x in KILL_PROB_INIT)
        ):
            mu, sigma = KILL_PROB_INIT
            return np.random.normal(mu, sigma, size=N_KILLER)
        else:
            raise ValueError("Please check your input for killing probability initialisation.")

def DeathFactor_ini(N_TARGET, DEATH_FAC_INIT=None):
    """
    Initialise death factor for all targets.

    - If DEATH_FAC_INIT is None: uniform random in [0.0, 0.1]
    - If DEATH_FAC_INIT is scalar: use as constant
    - If DEATH_FAC_INIT is (mu, sigma): normal distribution
    """
    if DEATH_FAC_INIT is None:
        return np.random.uniform(0.0, 0.1, size=N_TARGET)
    elif isinstance(DEATH_FAC_INIT, (int, float)):
        return np.full(N_TARGET, float(DEATH_FAC_INIT))
    elif (
        isinstance(DEATH_FAC_INIT, tuple)
        and len(DEATH_FAC_INIT) == 2
        and all(isinstance(x, (int, float)) for x in DEATH_FAC_INIT)
    ):
        mu, sigma = DEATH_FAC_INIT
        return np.random.normal(mu, sigma, size=N_TARGET)
    else:
        raise ValueError("Please check your input for death factor initialisation.")

def CellState_ini(N_CELL, STATE_INIT=1.0):
    """
    Initialise cell states for a population (values in [0, 1]).

    Modes
    -----
    - scalar (default=1.0): constant for all cells
    - None: uniform random in [0,1]
    - (mu, sigma): normal, clipped to [0,1]
    - list/tuple of sub-pops: [(ratio, state_spec), ...]
        where state_spec is scalar or (mu, sigma).
        Ratios can be any non-negative numbers; they'll be renormalised.
    """
    def _clip01(x): return np.clip(x, 0.0, 1.0)

    def _from_spec(spec, n):
        if isinstance(spec, (int, float)):
            return np.full(n, float(spec))
        if (isinstance(spec, tuple) and len(spec) == 2
            and all(isinstance(x, (int, float)) for x in spec)):
            mu, sigma = spec
            return _clip01(np.random.normal(mu, sigma, size=n))
        raise ValueError("Each sub-pop state must be a scalar or a (mu, sigma) tuple.")

    # Constant
    if isinstance(STATE_INIT, (int, float)):
        return np.full(N_CELL, float(STATE_INIT))

    # Uniform
    if STATE_INIT is None:
        return np.random.uniform(0.0, 1.0, size=N_CELL)

    # Single normal
    if (isinstance(STATE_INIT, tuple) and len(STATE_INIT) == 2
        and all(isinstance(x, (int, float)) for x in STATE_INIT)):
        mu, sigma = STATE_INIT
        return _clip01(np.random.normal(mu, sigma, size=N_CELL))

    # Mixture
    if isinstance(STATE_INIT, (list, tuple)) and len(STATE_INIT) > 0:
        # normalise input to (ratio, spec)
        subpops = []
        for item in STATE_INIT:
            if isinstance(item, dict):
                if "ratio" not in item or "state" not in item:
                    raise ValueError("Dict items must have 'ratio' and 'state' keys.")
                ratio, spec = item["ratio"], item["state"]
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                ratio, spec = item
            else:
                raise ValueError("Use (ratio, state) or {'ratio':..., 'state':...}.")
            if not isinstance(ratio, (int, float)) or ratio < 0:
                raise ValueError("Ratios must be non-negative numbers.")
            subpops.append((float(ratio), spec))

        total_ratio = sum(r for r, _ in subpops)
        if total_ratio <= 0:
            raise ValueError("Sum of ratios must be > 0.")
        subpops = [(r / total_ratio, spec) for r, spec in subpops]

        # convert ratios to counts (unbiased rounding)
        exact = np.array([r * N_CELL for r, _ in subpops], dtype=float)
        counts = np.floor(exact).astype(int)
        remainder = N_CELL - counts.sum()
        if remainder > 0:
            frac = exact - counts
            for i in np.argsort(-frac)[:remainder]:
                counts[i] += 1

        # build and shuffle
        chunks = [ _from_spec(spec, n) for (_, spec), n in zip(subpops, counts) if n > 0 ]
        if not chunks:
            return np.empty((0,), dtype=float)
        states = np.concatenate(chunks)
        np.random.shuffle(states)
        return _clip01(states)

    raise ValueError("STATE_INIT must be a scalar, None, (mu, sigma), or a list of (ratio, state).")


def update_cell_states(
    prev_states: np.ndarray,
    *,
    mode: str = "fatigue_on_contact",       # "constant", "fatigue_on_contact", "ou"
    dt: float = 0.0,
    new_contacts_count: np.ndarray | None = None,  # per-killer count of NEW contacts this step
    decay_per_contact: float = 0.05,  # fatigue decrement per NEW contact
    recovery_rate: float = 0.2,       # per-time recovery toward baseline
    baseline: float = 1.0,            # target baseline to recover toward
    noise_sigma: float = 0.0          # optional Gaussian noise per step
) -> np.ndarray:
    """
    Temporal update of killer cell states in [0,1].

    Modes
    -----
    - "constant": return prev_states unchanged.
    - "fatigue_on_contact": subtract decay_per_contact * (# new contacts), then recover toward baseline at 'recovery_rate', then add optional noise.
    - "ou": Ornstein–Uhlenbeck-like: dS = recovery_rate*(baseline - S)*dt + noise_sigma*sqrt(dt)*N(0,1).
            (ignores new_contacts_count)

    Notes
    -----
    - All outputs are clipped to [0, 1].
    - new_contacts_count should be a 1D array of length N_KILLER if used.
    """
    s = prev_states.astype(float).copy()

    if mode == "constant":
        return np.clip(s, 0.0, 1.0)

    if mode == "fatigue_on_contact":
        if new_contacts_count is None:
            new_contacts_count = np.zeros_like(s)
        # instantaneous fatigue for NEW contacts this step
        s -= decay_per_contact * new_contacts_count
        # continuous recovery toward baseline
        s += recovery_rate * (baseline - s) * dt
        # optional step noise
        if noise_sigma > 0.0:
            s += noise_sigma * np.sqrt(dt) * np.random.randn(*s.shape)
        return np.clip(s, 0.0, 1.0)

    if mode == "ou":
        # OU: ignore contacts; relax toward baseline with noise
        s += recovery_rate * (baseline - s) * dt
        if noise_sigma > 0.0:
            s += noise_sigma * np.sqrt(dt) * np.random.randn(*s.shape)
        return np.clip(s, 0.0, 1.0)

    raise ValueError("Unknown cell state mode. Use 'constant', 'fatigue_on_contact', or 'ou'.")

def adaptive_time_step(
    net_force: np.ndarray,
    DS: float,
    DT0: float,
    processed_time: float,
    SIM_DURATION: float
) -> float:
    """
    Compute the adaptive time-step based on the maximum net force.
    Returns the updated time-step (dt), which does not exceed DT0 or overshoot SIM_DURATION.
    """
    max_force = np.max(np.linalg.norm(net_force, axis=1))
    # Compute adaptive dt; do not overshoot the simulation duration
    if max_force > 0:
        dt_adapt = DS / max_force
    else:
        dt_adapt = DT0
    # dt must not exceed DT0 or the time left in the simulation
    return min(dt_adapt, DT0, SIM_DURATION - processed_time)

def get_killing_probabilities(N_KILLER, mode, value=None, mu_sigma=None, prev_probs=None, decay=0.0):
    """
    Returns the current killing probabilities for each killer cell.
    - mode: 'constant', 'normal', 'decay'
    - value: single float, for 'constant'
    - mu_sigma: (mean, std) tuple for 'normal'
    - prev_probs: previous step's probabilities for 'decay'
    - decay: decrease per contact (float)
    """
    if mode == 'constant':
        return np.full(N_KILLER, value)
    elif mode == 'normal':
        mu, sigma = mu_sigma
        return np.random.normal(mu, sigma, size=N_KILLER)
    elif mode == 'decay':
        # Should be called after updating prev_probs per contact
        return prev_probs
    else:
        raise ValueError("Unknown mode")
