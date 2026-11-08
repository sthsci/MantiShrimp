"""Stream the model's observations without accumulating cell trajectories in RAM."""

from contextlib import contextmanager, ExitStack
import csv
import gzip
import io

import numpy as np
from mantishrimp.simulation import EVENT_COLUMNS


@contextmanager
def history_sink(directory):
    with ExitStack() as files:
        def open_table(name):
            stream = gzip.GzipFile(directory / name, "wb", compresslevel=1, mtime=0)
            return files.enter_context(io.TextIOWrapper(stream, encoding="utf-8", newline=""))

        snapshots = open_table("snapshots.csv.gz")
        events = csv.DictWriter(open_table("events.csv.gz"), fieldnames=EVENT_COLUMNS)
        events.writeheader()
        steps = csv.writer(open_table("steps.csv.gz"))
        damage = csv.writer(open_table("pair_damage.csv.gz"))
        steps.writerow(("step", "time", "dt", "targets_alive", "bound_synapses", "productive_synapses"))
        damage.writerow(("step", "time", "killer_id", "target_id", "damage_increment", "cumulative_damage"))
        cells_writer = None

        def record(step, time, dt, cells, new_events, alive, bound, productive, rates, contributions):
            nonlocal cells_writer
            events.writerows(new_events)
            if cells:
                if cells_writer is None:
                    cells_writer = csv.DictWriter(snapshots, fieldnames=tuple(cells[0]))
                    cells_writer.writeheader()
                cells_writer.writerows(cells)
            steps.writerow((step, time, dt, alive.sum(), bound.sum(), productive.sum()))
            if rates is not None:
                # Unlisted pairs have zero increment and retain their last cumulative value.
                for k, t in zip(*np.nonzero(rates)):
                    damage.writerow((step, time, k, t, rates[k, t] * dt, contributions[k, t]))

        yield record
