"""Submit the Section 3 production grid from an Imperial login shell."""

import json
import os
from itertools import product
from pathlib import Path
import re
import shutil
import subprocess
import sys


def submit(bundle, batch, wrapper):
    if not shutil.which("qsub") or not wrapper.is_file():
        raise SystemExit("Run this on the Imperial login node with ~/bin/bsub available.")
    grid = json.loads((bundle / "section2_grid.json").read_text())
    if any(len(grid[k]) != 10 or len(set(grid[k])) != 10 for k in ("alpha", "sigma_mu", "sigma_r")):
        raise SystemExit("Expected the Section 2 grid: 10 distinct values on each axis.")
    subprocess.run(["bash", "run_sweep.sh", "--dry-run", "--workers", "8"], cwd=bundle, check=True)
    try:
        batch.mkdir()
    except FileExistsError:
        raise SystemExit(f"{batch} already exists. No new jobs submitted. Check qstat -u $USER.")
    print(f"Results: {batch}", flush=True)
    with (batch / "submitted.tsv").open("x", buffering=1) as log:
        for i, (alpha, mu) in enumerate(product(grid["alpha"], grid["sigma_mu"]), 1):
            tag = f"{i:03d}"
            result = subprocess.run([
                sys.executable, str(wrapper), "-n", f"s3g_{tag}", "-p", "8", "-m", "8", "-t", "08:00:00",
                "env", f"SWEEP_RUNS_DIR={batch / tag}", "bash", str(bundle / "run_sweep.sh"),
                "--alpha", str(alpha), "--sigma-mu", str(mu), "--cells", "1000", "--record-every", "1",
                "--workers", "8",
            ], cwd=bundle, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # The local bsub wrapper does not propagate qsub's exit code.
            job = re.search(r"(?m)^(\d+(?:\.[\w.-]+)?)\s*$", result.stdout)
            if result.returncode or not job:
                raise SystemExit(f"{result.stdout}\nStopped at slice {tag}: check qstat -u $USER before retrying.")
            print(tag, alpha, mu, job[1], sep="\t", file=log)
            print(f"[{i}/100] submitted {job[1]}", flush=True)


if __name__ == "__main__":
    ephemeral = os.environ.get("EPHEMERAL") or f"/rds/general/user/{os.environ['USER']}/ephemeral"
    submit(Path(__file__).resolve().parent, Path(ephemeral) / "section3_grid10", Path.home() / "bin/bsub")
