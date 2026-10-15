"""Legacy notebook interface to the shared engine in src/mantishrimp."""

import numpy as np

if __package__:
    from .assistant_function import (
        generate_two_populations, KillingProb_ini, DeathFactor_ini, CellState_ini,
        calculate_ij_forces,
    )
else:
    from assistant_function import (
        generate_two_populations, KillingProb_ini, DeathFactor_ini, CellState_ini,
        calculate_ij_forces,
    )

from mantishrimp import (
    CytotoxicityConfig, DomainConfig, MechanicsConfig, MotilityConfig,
    Simulation, SimulationConfig, simulate,
)


def KTSimulator(
    N_KILLER: int = 25,
    N_TARGET: int = 100,
    *,
    KILLER_RADIUS: float = 10.0,
    TARGET_RADIUS: float = 12.0,
    # Migration
    KILLER_POL_DECAY: float = 2.5,
    TARGET_POL_DECAY: float = 2.5,
    KILLER_POL_NOISE: float = 1,
    TARGET_POL_NOISE: float = 1,
    KILLER_MOTILITY: float = 150,
    TARGET_MOTILITY: float = 100,
    KILLER_TRANS_NOISE: float = 1,
    TARGET_TRANS_NOISE: float = 1,

    # LJ interaction
    #LJ_EPSILON_KK: float = 1,
    #LJ_EPSILON_TT: float = 1,
    #LJ_EPSILON_KT: float = 1000,

    # Hookean repulion
    KREP_KK: float = 1,
    KREP_TT: float = 1,
    KREP_KT: float = 1000,

    # Adhesion
    F_A_KK: float = 0,
    F_A_TT: float = 0,
    F_A_KT: float = 10,

    K_BIND: float = 0.5,
    K_UNBIND: float = 0.1,
    #R_CAP_FACTOR: float = 1.5,
    #CONTACT_TOL_FACTOR: float = 0.05,
    #UNBIND_DISTANCE_FACTOR: float = 1.75,
    
    RNG_SEED: int = None,

    # Simulation box
    BOX_X_MIN: float = -600,
    BOX_X_MAX: float = 600,
    BOX_Y_MIN: float = -600,
    BOX_Y_MAX: float = 600,
    boundary_condition: str = 'periodic',  # 'confined' or 'periodic'
    #Initial Spatial Distribution of Cells
    region_shape: str = "square",
    init_mode: str = "uniform",
    init_sigma: float = None,
    init_cell_bound: tuple = None,
    ini_cell_centre: tuple = None,
    ini_cell_region_radius: float = None,
    ini_seed: int = None,
    min_distance_factor: float = 2.5,
    # Time
    SIM_DURATION: float = 10,
    DT0: float = 1/10,
    DS: float = 40,
    
    # Killer initialisation
    KILL_PROBABILISTIC: bool = True,
    KILL_PROB_INIT: np.ndarray = None,
    kill_prob_mode: str = 'decay',  # 'constant'or 'decay'
    DECAY_PER_CONTACT: float = 0.05, # no worries this is for the killing probability decay.
    KILLER_STATE_INIT: float = 1.0,  # can be scalar, None, (mu, sigma), or list of (ratio, state)
    
    MIN_STATE_DECAY_dt: float = 0.001,
    MAX_STATE_DECAY_dt: float = 0.1,
    
    # Target initialisation
    DEATH_FAC_INIT: np.ndarray = None,
    KILL_DEATH_THRESHOLD: float = 1.0,
    
    TARGET_STATE_INIT: float = 1.0,  # can be scalar, None, (mu, sigma), or list of (ratio, state)
    
    RECOVERY: bool = True, 
    RECOVERY_SPEED: float = 0,
    DISAPPEAR_DELAY: float = 0.5,
    
):
    
    """Run the package engine and return the six legacy notebook outputs.

    Spatial and population initialisers retain the script interface. All time
    integration, binding, death and exhaustion are handled by Simulation.run.
    RECOVERY=False selects instant killing; True uses cumulative damage.
    F_A_KK, F_A_TT and DISAPPEAR_DELAY remain unused compatibility arguments.
    """
    arguments = locals().copy()
    for name, value in arguments.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if kill_prob_mode not in {"constant", "decay"}:
        raise ValueError("kill_prob_mode must be 'constant' or 'decay'")
    config = SimulationConfig(
        n_killers=N_KILLER, n_targets=N_TARGET,
        killer_radius=KILLER_RADIUS, target_radius=TARGET_RADIUS,
        duration=SIM_DURATION, max_dt=DT0, max_drift_displacement=DS,
        seed=RNG_SEED, initial_separation_factor=min_distance_factor,
        domain=DomainConfig(BOX_X_MIN, BOX_X_MAX, BOX_Y_MIN, BOX_Y_MAX, boundary_condition),
        motility=MotilityConfig(
            killer_polarity_decay=KILLER_POL_DECAY, target_polarity_decay=TARGET_POL_DECAY,
            killer_polarity_noise=KILLER_POL_NOISE, target_polarity_noise=TARGET_POL_NOISE,
            killer_speed=KILLER_MOTILITY, target_speed=TARGET_MOTILITY,
            killer_translation_noise=KILLER_TRANS_NOISE,
            target_translation_noise=TARGET_TRANS_NOISE,
        ),
        mechanics=MechanicsConfig(
            repulsion_kk=KREP_KK, repulsion_tt=KREP_TT, repulsion_kt=KREP_KT,
            adhesion_kt=F_A_KT, binding_rate=K_BIND, unbinding_rate=K_UNBIND,
        ),
        cytotoxicity=CytotoxicityConfig(
            death_threshold=KILL_DEATH_THRESHOLD,
            recovery_rate=RECOVERY_SPEED if RECOVERY else 0.0,
            instant_kill=not RECOVERY,
            kill_probability_decay_per_synapse=(
                DECAY_PER_CONTACT if kill_prob_mode == "decay" else 0.0
            ),
            exhaustion_rate_min=MIN_STATE_DECAY_dt, exhaustion_rate_max=MAX_STATE_DECAY_dt,
        ),
    )
    engine = Simulation(config)
    # The old initialisers use numpy's global RNG; seed them without changing caller state.
    random_state = np.random.get_state()
    try:
        np.random.seed(RNG_SEED)
        killer_positions, target_positions = generate_two_populations(
            N1=N_KILLER, N2=N_TARGET, radius1=KILLER_RADIUS, radius2=TARGET_RADIUS,
            shape=region_shape, mode=init_mode, sigma=init_sigma,
            bounds=init_cell_bound if init_cell_bound is not None else (
                0.9 * BOX_X_MIN, 0.9 * BOX_X_MAX, 0.9 * BOX_Y_MIN, 0.9 * BOX_Y_MAX
            ),
            centre=ini_cell_centre, region_radius=ini_cell_region_radius,
            seed=ini_seed if ini_seed is not None else RNG_SEED,
            min_distance_factor=min_distance_factor,
        )
        killer_state = CellState_ini(N_KILLER, KILLER_STATE_INIT)
        target_state = CellState_ini(N_TARGET, TARGET_STATE_INIT)
        probabilities = KillingProb_ini(N_KILLER, KILL_PROB_INIT, KILL_PROBABILISTIC)
        damage = DeathFactor_ini(N_TARGET, DEATH_FAC_INIT)
    finally:
        np.random.set_state(random_state)

    result = engine.run(
        initial_killer_positions=killer_positions, initial_target_positions=target_positions,
        initial_killer_state=killer_state, initial_target_state=target_state,
        initial_kill_probability=probabilities, initial_death_factor=damage,
    )
    history = result.snapshots.rename(columns={
        "polarity_x": "mu_x", "polarity_y": "mu_y", "state": "cell_state",
        "alive": "alive_status", "death_factor": "Death_Factor",
        "synapses": "contacts", "kill_probability": "killing_P",
    }).copy()
    history["killed_by"] = history.pop("primary_killer").map(
        lambda killer: [int(killer)] if killer is not None and np.isfinite(killer) else None
    )
    killers = history[history.cell_type == "killer"]
    targets = history[history.cell_type == "target"]
    killer_trajectory = killers[["x", "y"]].to_numpy().reshape(-1, N_KILLER, 2)
    target_trajectory = targets[["x", "y"]].to_numpy().reshape(-1, N_TARGET, 2)
    probability_history = list(killers.killing_P.to_numpy().reshape(-1, N_KILLER))
    damage_history = list(targets.Death_Factor.to_numpy().reshape(-1, N_TARGET))
    settings = {name: arguments[name] for name in (
        "KILLER_RADIUS", "TARGET_RADIUS", "BOX_X_MIN", "BOX_X_MAX", "BOX_Y_MIN", "BOX_Y_MAX",
        "boundary_condition", "KREP_KK", "KREP_TT", "KREP_KT", "F_A_KK", "F_A_TT", "F_A_KT",
    )}
    return history, killer_trajectory, target_trajectory, probability_history, damage_history, settings
