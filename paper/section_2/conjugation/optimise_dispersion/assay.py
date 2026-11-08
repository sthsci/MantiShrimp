"""The 100-minute dispersed assay from the adjacent adhesion sweep."""

from pathlib import Path
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parent / "src"
if not (SOURCE / "mantishrimp" / "__init__.py").is_file():
    raise SystemExit("Missing simulator: copy the entire optimise_dispersion folder.")
sys.path.insert(0, str(SOURCE))
from mantishrimp import (
    CytotoxicityConfig, MechanicsConfig, MotilityConfig, OutputConfig, Simulation, SimulationConfig,
)
from mantishrimp.simulation import _initial_positions


def make_config(task):
    return SimulationConfig(
        n_killers=task.n_killers, n_targets=task.n_targets, duration=task.duration,
        seed=task.seed, initial_separation_factor=2.5, max_dt=0.2,
        mechanics=MechanicsConfig(
            repulsion_kk=50, repulsion_tt=50, repulsion_kt=50,
            adhesion_kt=task.adhesion_kt, binding_rate=task.k_on, unbinding_rate=task.k_off,
        ),
        motility=MotilityConfig(
            killer_speed=50, target_speed=50, killer_polarity_decay=1, target_polarity_decay=5,
            killer_polarity_noise=task.sigma_mu, target_polarity_noise=0.2,
            killer_translation_noise=1, target_translation_noise=1,
        ),
        cytotoxicity=CytotoxicityConfig(killing_rate_max=1.5, exhaustion_rate_max=0.3),
        output=OutputConfig(record_every=sys.maxsize),
    )


def simulate_task(task):
    config = make_config(task)
    positions = _initial_positions(config, np.random.default_rng(task.spatial_seed))
    result = Simulation(config).run(initial_killer_positions=positions[:config.n_killers],
                                    initial_target_positions=positions[config.n_killers:])
    return result.summary()
