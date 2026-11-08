"""Load Hosty tracks without treating tracker predictions as observations."""

import json
from numbers import Integral, Real
from pathlib import Path
import re

import numpy as np
import pandas as pd


METADATA_COLUMNS = [
    "condition", "cell_type", "donor", "movie", "track_id", "track_uid",
    "start_frame", "n_positions", "n_valid_steps", "span_frames", "source",
]


def load_hosty(root, min_valid_steps=20, min_span_frames=12, pixel_size=1.0):
    """Return aligned ``(positions, metadata, audit)`` for all trajectory CSVs.

    Positions have shape (tracks, frames, 2), in x/y order, with missing frames
    and right padding represented by two NaNs. Each track starts at its first
    real observation, at time and position zero. ``span_frames`` is the elapsed
    number of frame intervals; ``n_valid_steps`` counts adjacent observed pairs.
    The filters apply to whole tracks, preserving internal gaps and dummy masks.

    Metadata and audit are DataFrames. Track IDs are local to a movie/channel;
    ``track_uid`` includes condition, movie, cell type and the local ID.
    ``pixel_size=1`` retains exported pixel coordinates; supply a measured
    length per pixel to convert them. No physical calibration is inferred.
    Source exports and their absolute, external metadata paths are never edited.
    """
    for name, value in (("min_valid_steps", min_valid_steps),
                        ("min_span_frames", min_span_frames)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if (isinstance(pixel_size, bool) or not isinstance(pixel_size, Real)
            or not np.isfinite(pixel_size) or pixel_size <= 0):
        raise ValueError("pixel_size must be finite and positive")

    root = Path(root)
    files = sorted(root.rglob("*_trajectories.csv"))
    if not files:
        raise FileNotFoundError(f"No trajectory CSVs in {root}")
    records, tracks, audit = [], [], []
    for path in files:
        source = path.relative_to(root).as_posix()
        condition, movie = path.parent.parent.name, path.parent.name
        match = re.fullmatch(r"(.+)_(NK_ch1|Tumour_ch2)_trajectories.csv", path.name)
        donor_match = re.match(r"Donor_(\d+)_", movie)
        if (not match or match[1] != movie or not donor_match
                or condition not in {"NT", "Bispecific", "Rtx"}):
            raise ValueError(f"Unrecognised trajectory identity: {source}")
        cell_type = match[2].split("_")[0]
        summary_path = path.with_name(path.name.replace("_trajectories.csv", "_summary.json"))
        summary = json.loads(summary_path.read_text())
        shape = summary.get("processed_shape_tyx", [])
        if (summary.get("condition") != condition
                or summary.get("channel_label") != cell_type
                or summary.get("segment_channel") != (1 if cell_type == "NK" else 2)
                or str(summary.get("file", "")).replace(" ", "_") != movie + ".nd2"
                or len(shape) != 3
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape)):
            raise ValueError(f"Inconsistent summary metadata: {source}")

        data = pd.read_csv(path, dtype={"dummy": str})
        required = {"id", "t", "x", "y", "dummy"}
        if not required.issubset(data.columns):
            raise ValueError(f"Missing columns {required - set(data.columns)}: {source}")
        for column in ("id", "t"):
            values = pd.to_numeric(data[column], errors="raise").to_numpy(dtype=float)
            if (not np.isfinite(values).all() or np.any(values < 0)
                    or np.any(values != np.floor(values))):
                raise ValueError(f"{column} must contain non-negative integers: {source}")
            data[column] = values.astype(np.int64)
        if data.duplicated(["id", "t"]).any():
            raise ValueError(f"Duplicate (id, t): {source}")
        if (data["t"] >= shape[0]).any():
            raise ValueError(f"Frame exceeds movie bounds: {source}")
        if not data["dummy"].isin(["True", "False"]).all():
            raise ValueError(f"dummy must be True or False: {source}")
        observed = data.loc[data["dummy"] == "False"].copy()
        coordinates = observed[["x", "y"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(coordinates).all():
            raise ValueError(f"Non-finite observed coordinates: {source}")
        observed[["x", "y"]] = coordinates
        n_tracks = data["id"].nunique()
        if summary.get("n_tracks") != n_tracks:
            raise ValueError(f"Track count differs from summary: {source}")
        retained = 0
        for track_id, track in observed.sort_values(["id", "t"]).groupby("id", sort=False):
            times = track["t"].to_numpy()
            span = int(times[-1] - times[0])
            n_valid = int(np.count_nonzero(np.diff(times) == 1))
            if n_valid < min_valid_steps or span < min_span_frames:
                continue
            xy = track[["x", "y"]].to_numpy(dtype=float)
            positions = np.full((span + 1, 2), np.nan)
            positions[times - times[0]] = (xy - xy[0]) * pixel_size
            tracks.append(positions)
            records.append({
                "condition": condition, "cell_type": cell_type,
                "donor": int(donor_match[1]), "movie": movie, "track_id": int(track_id),
                "track_uid": f"{condition}|{movie}|{cell_type}|{track_id}",
                "start_frame": int(times[0]), "n_positions": len(times),
                "n_valid_steps": n_valid, "span_frames": span, "source": source,
            })
            retained += 1
        audit.append({
            "condition": condition, "cell_type": cell_type,
            "donor": int(donor_match[1]), "movie": movie, "source": source,
            "movie_frames": shape[0], "rows": len(data), "observed_rows": len(observed),
            "dummy_rows": len(data) - len(observed), "tracks": n_tracks,
            "retained_tracks": retained, "dropped_tracks": n_tracks - retained,
        })
    positions = np.full((len(tracks), max((len(p) for p in tracks), default=0), 2), np.nan)
    for index, track in enumerate(tracks):
        positions[index, :len(track)] = track
    return positions, pd.DataFrame(records, columns=METADATA_COLUMNS), pd.DataFrame(audit)


def self_check():
    """Check gaps, dummy exclusion, identity, filtering and input validation."""
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = []
        for movie, cell_type in (("Donor_1_01_A", "NK"), ("Donor_1_01_A", "Tumour"),
                                 ("Donor_1_02_B", "NK")):
            channel = 1 if cell_type == "NK" else 2
            folder = root / "NT" / movie
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{movie}_{cell_type}_ch{channel}_trajectories.csv"
            path.write_text("id,t,x,y,dummy\n1,2,10,20,False\n1,3,11,22,False\n"
                            "1,4,999,999,True\n1,5,13,26,False\n1,7,15,30,False\n"
                            "1,8,16,32,False\n2,0,0,0,False\n")
            path.with_name(path.name.replace("_trajectories.csv", "_summary.json")).write_text(
                json.dumps({"condition": "NT", "channel_label": cell_type,
                            "segment_channel": channel, "file": movie.replace("_A", " A") + ".nd2",
                            "processed_shape_tyx": [10, 512, 512], "n_tracks": 2}))
            paths.append(path)
        positions, metadata, audit = load_hosty(root, 2, 6, pixel_size=0.5)
        assert positions.shape == (3, 7, 2)
        assert metadata["track_uid"].nunique() == 3
        assert (metadata["start_frame"] == 2).all()
        assert (metadata["n_positions"] == 5).all()
        assert (metadata["n_valid_steps"] == 2).all()
        assert np.allclose(positions[:, 0], 0)
        assert np.allclose(positions[:, -1], [3, 6])
        assert np.isnan(positions[:, [2, 4]]).all()
        assert (audit["dummy_rows"] == 1).all() and (audit["dropped_tracks"] == 1).all()
        for row, position in zip(metadata.itertuples(), positions):
            assert np.isfinite(position[:, 0]).sum() == row.n_positions
            assert row.cell_type in row.source and row.movie in row.source
        empty, metadata, audit = load_hosty(root, 3, 6)
        assert empty.shape == (0, 0, 2) and metadata.empty
        assert (audit["dropped_tracks"] == 2).all()
        original = paths[0].read_text()
        for invalid, message in ((original + "1,2,0,0,False\n", "Duplicate"),
                                 (original.replace("10,20,False", "nan,20,False"), "Non-finite"),
                                 (original.replace("1,8,", "1,10,"), "Frame exceeds"),
                                 (original.replace("999,999,True", "999,999,unknown"), "dummy")):
            paths[0].write_text(invalid)
            try:
                load_hosty(root, 0, 0)
            except ValueError as error:
                assert message in str(error), str(error)
            else:
                raise AssertionError(f"Expected rejection: {message}")
        paths[0].write_text(original)
    print("Hosty loader self-check passed.")


if __name__ == "__main__":
    self_check()
