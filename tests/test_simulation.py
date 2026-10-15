from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize("target_alive", [True, False])
def test_bound_adhesion_adds_to_overlap_repulsion(target_alive):
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
            if not target_alive and not bound:
                expected = 0.0
            forces = _forces(
                np.array([[distance, 0.0], [0.0, 0.0]]),
                np.array([True, target_alive]), np.array([[bound]]), config,
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
    assert "synapse_ended" not in set(result.events["event"])


@pytest.fixture
def conjugation_config():
    return SimulationConfig(
        n_killers=1, n_targets=1, duration=4.0, max_dt=0.1, seed=3,
        motility=ZERO_MOTILITY,
        mechanics=MechanicsConfig(
            repulsion_kk=0.0, repulsion_tt=0.0, repulsion_kt=0.0,
            adhesion_kt=0.0, binding_rate=1000.0, unbinding_rate=0.0,
        ),
        cytotoxicity=CytotoxicityConfig(
            killing_rate_max=5.0, death_threshold=0.2,
            initial_target_state=0.5, recovery_rate=1.0,
            exhaustion_rate_min=0.0, exhaustion_rate_max=0.5,
        ),
    )


@pytest.mark.parametrize("unbinding_rate", [0.0, 1.0, 1e6])
@pytest.mark.parametrize("exhaustion_rate", [0.0, 0.5])
def test_dead_target_exhausts_killer_until_unbinding(
    conjugation_config, unbinding_rate, exhaustion_rate,
):
    config = replace(
        conjugation_config,
        mechanics=replace(conjugation_config.mechanics, unbinding_rate=unbinding_rate),
        cytotoxicity=replace(
            conjugation_config.cytotoxicity, exhaustion_rate_max=exhaustion_rate,
        ),
    )
    result = Simulation(config).run(
        initial_killer_positions=np.array([[0.0, 0.0]]),
        initial_target_positions=np.array([[22.0, 0.0]]),
    )
    deaths = result.events[result.events["event"] == "target_died"]
    ends = result.events[result.events["event"] == "synapse_ended"]
    assert len(deaths) == 1
    assert result.kills_per_cell().tolist() == [1]
    assert result.synapses_per_cell().tolist() == [1]  # No rebinding to the dead target.
    if unbinding_rate == 0:
        assert ends.empty
        release_time = config.duration
    else:
        assert len(ends) == 1
        assert ends.iloc[0]["reason"] == "stochastic"
        release_time = ends.iloc[0]["time"]
        assert deaths.iloc[0]["time"] <= release_time < config.duration
        if unbinding_rate == 1.0:
            assert release_time > deaths.iloc[0]["time"]

    killers = result.snapshots[result.snapshots["cell_type"] == "killer"]
    active_dt = np.diff(np.minimum(killers["time"].to_numpy(), release_time))
    expected_state = np.r_[1.0, np.cumprod(1 - exhaustion_rate * 0.5 * active_dt)]
    np.testing.assert_allclose(killers["state"], expected_state)
    dead_targets = result.snapshots[
        (result.snapshots["cell_type"] == "target") & ~result.snapshots["alive"]
    ]
    np.testing.assert_allclose(dead_targets["death_factor"], config.cytotoxicity.death_threshold)


def test_nonproductive_bond_to_dead_target_does_not_exhaust(conjugation_config):
    config = replace(
        conjugation_config, n_killers=2,
        cytotoxicity=replace(conjugation_config.cytotoxicity, kill_probability=0.5),
    )
    result = Simulation(config).run(
        initial_killer_positions=np.array([[-22.0, 0.0], [22.0, 0.0]]),
        initial_target_positions=np.array([[0.0, 0.0]]),
    )
    formed = result.events[result.events["event"] == "synapse_formed"]
    assert formed["value"].tolist() == [1.0, 0.0]
    assert "synapse_ended" not in set(result.events["event"])
    assert result.kills_per_cell().tolist() == [1, 0]
    killers = result.snapshots[result.snapshots["cell_type"] == "killer"]
    productive = killers[killers["cell_id"] == 0]
    assert np.all(np.diff(productive["state"]) < 0)
    np.testing.assert_allclose(killers[killers["cell_id"] == 1]["state"], 1.0)


def test_dead_target_release_respects_drift_limit(conjugation_config):
    config = replace(
        conjugation_config, n_targets=2, killer_radius=0.5, target_radius=0.5,
        duration=1.0, max_drift_displacement=0.01,
        mechanics=replace(
            conjugation_config.mechanics, repulsion_kt=10.0,
            binding_rate=1e6, unbinding_rate=5.0,
        ),
        cytotoxicity=replace(conjugation_config.cytotoxicity, death_threshold=1e-6),
    )
    result = Simulation(config).run(
        initial_killer_positions=np.array([[0.0, 0.0]]),
        initial_target_positions=np.array([[-0.2, 0.0], [0.2, 0.0]]),
    )
    assert result.kills_per_cell().tolist() == [2]
    ends = result.events[result.events["event"] == "synapse_ended"]
    assert len(ends) == 2
    assert (ends["reason"] == "stochastic").all()
    for _, snapshots in result.snapshots.groupby(["cell_type", "cell_id"]):
        displacement = np.diff(snapshots[["x", "y"]].to_numpy(), axis=0)
        assert np.all(
            np.linalg.norm(displacement, axis=1)
            <= config.max_drift_displacement + 1e-12
        )
        dead = snapshots[~snapshots["alive"]]
        if not dead.empty:
            np.testing.assert_allclose(np.diff(dead[["x", "y"]], axis=0), 0.0)


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


@pytest.mark.parametrize("instant", [False, True])
def test_instant_kill_preserves_bonds_even_without_cumulative_damage(conjugation_config, instant):
    config = replace(
        conjugation_config, duration=0.3,
        cytotoxicity=CytotoxicityConfig(
            killing_rate_max=0.0, initial_target_state=0.0, instant_kill=instant,
            exhaustion_rate_min=0.1, exhaustion_rate_max=0.1,
        ),
    )
    result = Simulation(config).run(
        initial_killer_positions=np.array([[0.0, 0.0]]),
        initial_target_positions=np.array([[22.0, 0.0]]),
    )
    targets = result.snapshots[result.snapshots.cell_type == "target"]
    assert targets.alive.iloc[1:].eq(not instant).all()
    assert targets.synapses.iloc[1:].tolist() == [[0]] * (len(targets) - 1)
    assert result.kills_per_cell().tolist() == [int(instant)]
    if instant:
        assert targets.primary_killer.iloc[1:].eq(0).all()
        assert targets.death_factor.iloc[1:].eq(config.cytotoxicity.death_threshold).all()
    killers = result.snapshots[result.snapshots.cell_type == "killer"]
    assert np.all(np.diff(killers.state) < 0)


@pytest.mark.parametrize("name, values", [
    ("initial_killer_state", [0.5, 0.5]),
    ("initial_target_state", [np.nan]),
    ("initial_kill_probability", [1.1]),
    ("initial_death_factor", [-0.1]),
    ("initial_death_factor", [1.0]),
    ("initial_killer_positions", [[np.inf, 0.0]]),
])
def test_explicit_initial_arrays_are_validated(conjugation_config, name, values):
    initial = dict(
        initial_killer_positions=np.array([[0.0, 0.0]]),
        initial_target_positions=np.array([[22.0, 0.0]]),
    )
    initial[name] = np.array(values)
    with pytest.raises(ValueError):
        Simulation(conjugation_config).run(**initial)


def test_run_does_not_mutate_initial_arrays(conjugation_config):
    initial = dict(
        initial_killer_positions=np.array([[0.0, 0.0]]),
        initial_target_positions=np.array([[22.0, 0.0]]),
        initial_killer_state=np.array([0.8]), initial_target_state=np.array([0.5]),
        initial_death_factor=np.array([0.0]), initial_kill_probability=np.array([1.0]),
    )
    saved = {key: value.copy() for key, value in initial.items()}
    Simulation(conjugation_config).run(**initial)
    for key in initial:
        np.testing.assert_array_equal(initial[key], saved[key])
