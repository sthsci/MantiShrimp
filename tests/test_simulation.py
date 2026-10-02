from dataclasses import replace

import numpy as np
import pandas as pd

from mantishrimp import (
    CytotoxicityConfig,
    DomainConfig,
    MechanicsConfig,
    MotilityConfig,
    Simulation,
    SimulationConfig,
)
from mantishrimp.simulation import _forces, _minimum_image


ZERO_MOTILITY = MotilityConfig(
    killer_polarity_decay=0.0,
    target_polarity_decay=0.0,
    killer_polarity_noise=0.0,
    target_polarity_noise=0.0,
    killer_speed=0.0,
    target_speed=0.0,
    killer_translation_noise=0.0,
    target_translation_noise=0.0,
)


def test_minimum_image_geometry_crosses_periodic_boundary():
    domain = DomainConfig(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0)
    displacement = _minimum_image(np.array([98.0, -97.0]), domain)
    np.testing.assert_allclose(displacement, [-2.0, 3.0])


def test_bound_adhesion_adds_to_overlap_repulsion():
    config = SimulationConfig(
        n_killers=1, n_targets=1, killer_radius=0.5, target_radius=0.5,
        mechanics=MechanicsConfig(repulsion_kt=10.0, adhesion_kt=2.0),
    )
    # L=1, Rcap=1.5; the bound equilibrium is L - F_A/Krep = 0.8.
    for distance, unbound_force, bound_force in [
        (0.0, 0.0, 0.0),  # coincident centres retain the existing zero guard
        (0.5, 5.0, 3.0),
        (0.8, 2.0, 0.0),
        (0.9, 1.0, -1.0),
        (1.0, 0.0, -2.0),
        (1.5 - 1e-8, 0.0, -2.0),
        (1.5, 0.0, 0.0),
        (1.5 + 1e-8, 0.0, 0.0),
    ]:
        for bound, expected in [(False, unbound_force), (True, bound_force)]:
            forces = _forces(
                np.array([[distance, 0.0], [0.0, 0.0]]),
                np.array([True, True]), np.array([[bound]]), config,
            )
            np.testing.assert_allclose(
                forces, [[expected, 0.0], [-expected, 0.0]], atol=1e-12,
            )


def test_unbinding_at_force_balance_respects_drift_limit():
    config = SimulationConfig(
        n_killers=1, n_targets=1, killer_radius=0.5, target_radius=0.5,
        duration=0.015, max_dt=0.1, max_drift_displacement=0.01, seed=3,
        motility=ZERO_MOTILITY,
        mechanics=MechanicsConfig(
            repulsion_kt=10.0, adhesion_kt=2.0,
            binding_rate=1e6, unbinding_rate=1e6,
        ),
        cytotoxicity=CytotoxicityConfig(killing_rate_max=0.0),
    )
    result = Simulation(config).run(
        initial_killer_positions=np.array([[0.8, 0.0]]),
        initial_target_positions=np.array([[0.0, 0.0]]),
    )
    assert {"synapse_formed", "synapse_ended"}.issubset(result.events["event"])
    for _, snapshots in result.snapshots.groupby("cell_type"):
        displacement = np.diff(snapshots[["x", "y"]].to_numpy(), axis=0)
        assert np.all(
            np.linalg.norm(displacement, axis=1)
            <= config.max_drift_displacement + 1e-12
        )


def test_contact_synapse_and_kill_are_distinct_events():
    config = SimulationConfig(
        n_killers=1,
        n_targets=1,
        duration=0.2,
        max_dt=0.1,
        max_drift_displacement=1.0,
        seed=3,
        domain=DomainConfig(x_min=-100, x_max=100, y_min=-100, y_max=100),
        motility=ZERO_MOTILITY,
        mechanics=MechanicsConfig(
            repulsion_kk=0.0,
            repulsion_tt=0.0,
            repulsion_kt=0.0,
            adhesion_kt=0.0,
            binding_rate=1000.0,
            unbinding_rate=0.0,
        ),
        cytotoxicity=CytotoxicityConfig(
            killing_rate_max=5.0,
            death_threshold=0.2,
            exhaustion_rate_min=0.0,
            exhaustion_rate_max=0.0,
        ),
    )
    result = Simulation(config).run(
        initial_killer_positions=np.array([[0.0, 0.0]]),
        initial_target_positions=np.array([[22.0, 0.0]]),
    )

    assert result.contacts_per_cell().tolist() == [1]
    assert result.synapses_per_cell().tolist() == [1]
    assert result.kills_per_cell().tolist() == [1]
    assert {
        "contact_started",
        "synapse_formed",
        "target_killed",
        "target_died",
    }.issubset(set(result.events["event"]))
    assert result.summary()["targets_killed"] == 1


def test_zero_event_cells_are_retained():
    config = SimulationConfig(
        n_killers=2,
        n_targets=1,
        duration=0.1,
        max_dt=0.1,
        seed=1,
        domain=DomainConfig(x_min=-200, x_max=200, y_min=-200, y_max=200),
        motility=ZERO_MOTILITY,
        mechanics=replace(MechanicsConfig(), binding_rate=0.0),
    )
    result = Simulation(config).run(
        initial_killer_positions=np.array([[-100.0, 0.0], [100.0, 0.0]]),
        initial_target_positions=np.array([[0.0, 100.0]]),
    )

    np.testing.assert_array_equal(result.contacts_per_cell(), [0, 0])
    np.testing.assert_array_equal(result.kills_per_cell(), [0, 0])


def test_seed_reproduces_initialisation_and_dynamics():
    config = SimulationConfig(
        n_killers=2,
        n_targets=3,
        duration=0.1,
        max_dt=0.05,
        seed=11,
        domain=DomainConfig(x_min=-100, x_max=100, y_min=-100, y_max=100),
    )
    first = Simulation(config).run()
    second = Simulation(config).run()

    pd.testing.assert_frame_equal(first.snapshots, second.snapshots)
    pd.testing.assert_frame_equal(first.events, second.events)


def test_result_can_be_saved(tmp_path):
    config = SimulationConfig(
        n_killers=1,
        n_targets=1,
        duration=0.1,
        max_dt=0.1,
        seed=5,
        domain=DomainConfig(x_min=-100, x_max=100, y_min=-100, y_max=100),
    )
    output = Simulation(config).run().save(tmp_path / "run")

    assert (output / "snapshots.csv").is_file()
    assert (output / "events.csv").is_file()
    assert (output / "final_cells.csv").is_file()
    assert (output / "config.json").is_file()
