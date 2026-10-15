import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def legacy(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "script" / "simulator"))
    return importlib.import_module("simulator")


@pytest.fixture
def run_conjugation(legacy, monkeypatch):
    def fixed_positions(**kwargs):
        return (
            np.tile([0.0, 0.0], (kwargs["N1"], 1)),
            np.tile([22.0, 0.0], (kwargs["N2"], 1)),
        )

    monkeypatch.setattr(legacy, "generate_two_populations", fixed_positions)

    def run(**overrides):
        parameters = dict(
            N_KILLER=1, N_TARGET=1, SIM_DURATION=2.0, DT0=0.125,
            KILLER_POL_NOISE=0.0, TARGET_POL_NOISE=0.0,
            KILLER_MOTILITY=0.0, TARGET_MOTILITY=0.0,
            KILLER_TRANS_NOISE=0.0, TARGET_TRANS_NOISE=0.0,
            KREP_KK=0.0, KREP_TT=0.0, KREP_KT=0.0, F_A_KT=0.0,
            K_BIND=1e6, K_UNBIND=0.0, RNG_SEED=3,
            KILL_PROB_INIT=1.0, DECAY_PER_CONTACT=1.0,
            DEATH_FAC_INIT=0.0, KILL_DEATH_THRESHOLD=0.02,
            TARGET_STATE_INIT=0.5, MIN_STATE_DECAY_dt=0.0,
            MAX_STATE_DECAY_dt=0.5,
        )
        parameters.update(overrides)
        return legacy.KTSimulator(**parameters)

    return run


@pytest.mark.parametrize("recovery", [True, False])
@pytest.mark.parametrize("unbind_rate", [0.0, 2.0, 1e6])
@pytest.mark.parametrize("exhaustion_rate", [0.0, 0.5])
def test_script_retains_dead_conjugation_until_unbinding(
    run_conjugation, recovery, unbind_rate, exhaustion_rate,
):
    output = run_conjugation(
        RECOVERY=recovery, K_UNBIND=unbind_rate, MAX_STATE_DECAY_dt=exhaustion_rate,
    )
    assert len(output) == 6
    history, killer_positions, target_positions, probabilities, damage, settings = output
    assert set(history.columns) == {
        "step", "time", "cell_type", "cell_id", "x", "y", "mu_x", "mu_y",
        "cell_state", "alive_status", "killing_P", "contacts", "Death_Factor", "killed_by",
    }
    killers = history[history.cell_type == "killer"]
    targets = history[history.cell_type == "target"]
    bound = killers.contacts.map(bool).to_numpy()
    assert targets.alive_status.tolist() == [True] + [False] * (len(targets) - 1)
    assert bound[1]
    if unbind_rate == 0:
        assert bound[1:].all()
    else:
        first_release = np.flatnonzero(~bound & (killers.step.to_numpy() > 0))[0]
        assert first_release > 1
        assert not bound[first_release:].any()
        if unbind_rate == 2.0:
            assert first_release > 2
    np.testing.assert_array_equal(targets.contacts.map(bool), bound)
    expected_state = np.cumprod(1.0 - exhaustion_rate * 0.5 * 0.125 * bound)
    np.testing.assert_allclose(killers.cell_state, expected_state)
    np.testing.assert_allclose(targets.Death_Factor.iloc[1:], 0.02)
    assert targets.killed_by.iloc[1:].tolist() == [[0]] * (len(targets) - 1)
    assert killers.killing_P.iloc[1:].eq(0.0).all()
    assert killer_positions.shape == target_positions.shape == (len(killers), 1, 2)
    np.testing.assert_allclose(np.diff(target_positions, axis=0), 0.0)
    assert len(probabilities) == len(damage) == len(killers)
    assert settings["KILLER_RADIUS"] == 10.0


@pytest.mark.parametrize("recovery", [True, False])
@pytest.mark.parametrize("second_productive", [False, True])
def test_each_productive_killer_exhausts_after_death(
    legacy, run_conjugation, monkeypatch, recovery, second_productive,
):
    monkeypatch.setattr(
        legacy, "KillingProb_ini", lambda *args: np.array([1.0, float(second_productive)]),
    )
    history, *_ = run_conjugation(N_KILLER=2, RECOVERY=recovery)
    killers = history[history.cell_type == "killer"]
    for killer_id, productive in enumerate([True, second_productive]):
        snapshots = killers[killers.cell_id == killer_id]
        assert snapshots.contacts.iloc[1:].tolist() == [[0]] * (len(snapshots) - 1)
        expected = (1 - 0.5 * 0.5 * 0.125 * productive) ** snapshots.step.to_numpy()
        np.testing.assert_allclose(snapshots.cell_state, expected)
    targets = history[history.cell_type == "target"]
    # The shared engine assigns each death to one primary killer, including ties.
    credit = [0]
    assert targets.killed_by.iloc[1:].tolist() == [credit] * (len(targets) - 1)


@pytest.mark.parametrize(
    "initially_bound, unbind_rate, distance, expected_bound, expected_force",
    [
        (True, 0.0, 0.5, True, 3.0),
        (True, 0.0, 1.0, True, -2.0),
        (True, 1e6, 0.5, False, 0.0),
        (False, 0.0, 0.5, False, 0.0),
        (True, 0.0, 1.6, False, 0.0),
    ],
)
def test_helper_handles_only_existing_dead_target_bonds(
    legacy, initially_bound, unbind_rate, distance, expected_bound, expected_force,
):
    force, bound = legacy.calculate_ij_forces(
        1, 1, killer_positions=np.array([[distance, 0.0]]),
        target_positions=np.array([[0.0, 0.0]]), killer_alive=np.array([True]),
        target_alive=np.array([False]), KREP_KK=0.0, KREP_TT=0.0, KREP_KT=10.0,
        F_A_KK=0.0, F_A_TT=0.0, F_A_KT=2.0,
        KILLER_RADIUS=0.5, TARGET_RADIUS=0.5, is_bound=np.array([[initially_bound]]),
        rng=np.random.default_rng(3), k_bind=1e6, k_unbind=unbind_rate, dt=0.125,
    )
    assert bound.item() == expected_bound
    np.testing.assert_allclose(force[0, 1], [expected_force, 0.0])
    np.testing.assert_allclose(force[1, 0], [-expected_force, 0.0])


@pytest.mark.parametrize("boundary", ["periodic", "confined"])
@pytest.mark.parametrize("instant", [False, True])
def test_legacy_and_package_trajectories_are_identical(legacy, monkeypatch, boundary, instant):
    from mantishrimp import (
        CytotoxicityConfig, DomainConfig, MechanicsConfig, MotilityConfig,
        Simulation, SimulationConfig, simulate,
    )

    positions = np.array([[-14.375, 0.0], [14.375, 0.0]]) if boundary == "periodic" else np.array([
        [-0.625, 0.0], [0.625, 0.0],
    ])
    monkeypatch.setattr(legacy, "generate_two_populations", lambda **kwargs: (
        positions[:1].copy(), positions[1:].copy(),
    ))
    config = SimulationConfig(
        n_killers=1, n_targets=1, killer_radius=0.5, target_radius=0.75,
        duration=0.75, max_dt=0.08, max_drift_displacement=0.04, seed=17,
        domain=DomainConfig(-15, 15, -15, 15, boundary),
        motility=MotilityConfig(1.1, 1.3, 0.2, 0.4, 2.0, 3.0, 0.1, 0.3),
        mechanics=MechanicsConfig(3.0, 4.0, 5.0, 0.2, 1e6, 1.0),
        cytotoxicity=CytotoxicityConfig(
            death_threshold=0.15, recovery_rate=0.0 if instant else 0.1,
            instant_kill=instant, kill_probability_decay_per_synapse=0.2,
            exhaustion_rate_min=0.05, exhaustion_rate_max=0.3,
        ),
    )
    result = Simulation(config).run(
        initial_killer_positions=positions[:1], initial_target_positions=positions[1:],
        initial_killer_state=np.array([0.8]), initial_target_state=np.array([0.6]),
        initial_kill_probability=np.array([1.0]), initial_death_factor=np.array([0.02]),
    )
    output = legacy.KTSimulator(
        N_KILLER=1, N_TARGET=1, KILLER_RADIUS=0.5, TARGET_RADIUS=0.75,
        SIM_DURATION=0.75, DT0=0.08, DS=0.04, RNG_SEED=17,
        BOX_X_MIN=-15, BOX_X_MAX=15, BOX_Y_MIN=-15, BOX_Y_MAX=15,
        boundary_condition=boundary, KILLER_POL_DECAY=1.1, TARGET_POL_DECAY=1.3,
        KILLER_POL_NOISE=0.2, TARGET_POL_NOISE=0.4,
        KILLER_MOTILITY=2.0, TARGET_MOTILITY=3.0,
        KILLER_TRANS_NOISE=0.1, TARGET_TRANS_NOISE=0.3,
        KREP_KK=3.0, KREP_TT=4.0, KREP_KT=5.0, F_A_KT=0.2,
        K_BIND=1e6, K_UNBIND=1.0, RECOVERY=not instant, RECOVERY_SPEED=0.1,
        KILL_DEATH_THRESHOLD=0.15, DECAY_PER_CONTACT=0.2,
        MIN_STATE_DECAY_dt=0.05, MAX_STATE_DECAY_dt=0.3,
        KILLER_STATE_INIT=0.8, TARGET_STATE_INIT=0.6,
        KILL_PROB_INIT=1.0, DEATH_FAC_INIT=0.02,
    )
    history, killer_positions, target_positions, probabilities, damage, _ = output
    for old, new in {
        "step": "step", "time": "time", "x": "x", "y": "y",
        "mu_x": "polarity_x", "mu_y": "polarity_y", "cell_state": "state",
        "alive_status": "alive", "Death_Factor": "death_factor", "killing_P": "kill_probability",
    }.items():
        np.testing.assert_array_equal(history[old], result.snapshots[new])
    assert history.contacts.tolist() == result.snapshots.synapses.tolist()
    for kind, trajectory in (("killer", killer_positions), ("target", target_positions)):
        cells = result.snapshots[result.snapshots.cell_type == kind]
        np.testing.assert_array_equal(trajectory, cells[["x", "y"]].to_numpy().reshape(-1, 1, 2))
    np.testing.assert_array_equal(probabilities, history[history.cell_type == "killer"].killing_P.to_numpy()[:, None])
    np.testing.assert_array_equal(damage, history[history.cell_type == "target"].Death_Factor.to_numpy()[:, None])
    for event in result.events[result.events.event == "target_killed"].itertuples():
        death = history[(history.cell_type == "target") & (history.step == event.step)]
        assert death.killed_by.iloc[0] == [int(event.killer_id)]
    assert legacy.Simulation is Simulation
    assert legacy.simulate is simulate


def test_script_initialisation_is_reproducible_and_preserves_global_rng(legacy):
    import pandas as pd

    arguments = dict(
        N_KILLER=2, N_TARGET=3, SIM_DURATION=0.12, DT0=0.04, RNG_SEED=42,
        BOX_X_MIN=-100, BOX_X_MAX=100, BOX_Y_MIN=-100, BOX_Y_MAX=100,
        region_shape="circle", init_mode="gaussian", ini_cell_centre=(0, 0),
        ini_cell_region_radius=85, min_distance_factor=1.1,
        KILLER_STATE_INIT=[(0.5, 0.25), (0.5, 0.75)], TARGET_STATE_INIT=None,
        KILL_PROB_INIT=0.8, kill_prob_mode="constant", DEATH_FAC_INIT=0.0,
    )
    np.random.seed(21)
    saved = np.random.get_state()
    first = legacy.KTSimulator(**arguments)
    np.testing.assert_array_equal(saved[1], np.random.get_state()[1])
    assert saved[2:] == np.random.get_state()[2:]
    np.random.seed(987)
    second = legacy.KTSimulator(**arguments)
    pd.testing.assert_frame_equal(first[0], second[0])
    for trajectory in first[1:3]:
        assert (np.linalg.norm(trajectory[0], axis=1) <= 85).all()
    initial = first[0][(first[0].step == 0) & (first[0].cell_type == "killer")]
    assert sorted(initial.cell_state) == [0.25, 0.75]
    assert first[0].killed_by.isna().all()
    np.testing.assert_array_equal(first[3], np.full((len(first[3]), 2), 0.8))


@pytest.mark.parametrize("invalid", [
    {"DT0": 0}, {"DT0": np.nan}, {"boundary_condition": "typo"},
    {"kill_prob_mode": "typo"}, {"KILL_PROB_INIT": 1.5}, {"TARGET_STATE_INIT": np.nan},
])
def test_legacy_rejects_invalid_parameters(run_conjugation, invalid):
    with pytest.raises(ValueError):
        run_conjugation(**invalid)


def test_import_from_simulator_directory_without_installation():
    import os
    import subprocess
    import sys

    directory = Path(__file__).resolve().parents[1] / "script" / "simulator"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    subprocess.run([
        sys.executable, "-c",
        "from simulator import simulate; from mantishrimp import simulate as core; "
        "assert simulate is core; "
        f"assert core.__code__.co_filename.startswith({str(directory.parents[1] / 'src')!r})",
    ], cwd=directory, env=environment, check=True, capture_output=True, text=True)
