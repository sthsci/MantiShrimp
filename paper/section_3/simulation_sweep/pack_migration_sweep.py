"""Build the code-only HPC archive beside the sweep, without macOS extended headers."""

import hashlib
from pathlib import Path
import tarfile
import tempfile


def pack():
    root = Path(__file__).resolve().parent / "migration_sweep"
    files = [root / name for name in ("LICENSE", "README.md", "requirements.txt", "run_sweep.py",
             "run_sweep.sh", "section2_grid.json", "submit_full_sweep.py", "submit_full_sweep.sh", "test_run_sweep.py", "timeline.py")]
    files.extend(sorted((root / "src/mantishrimp").glob("*.py")))
    payload = {p.relative_to(root).as_posix(): p for p in files}
    sums = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in payload.items()}
    target = root.parent / "migration_sweep-hpc.tar.gz"
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
        inventory = Path(tmp) / "SHA256SUMS"
        inventory.write_text("".join(f"{digest}  {name}\n" for name, digest in sums.items()), encoding="ascii")
        payload[inventory.name] = inventory
        bundle = Path(tmp) / target.name
        with tarfile.open(bundle, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
            for name, path in payload.items():
                archive.add(path, arcname=f"{root.name}/{name}", recursive=False)
        with tarfile.open(bundle) as archive:
            assert set(archive.getnames()) == {f"{root.name}/{n}" for n in payload}
            for name, path in payload.items():
                member = archive.getmember(f"{root.name}/{name}")
                assert member.isfile() and not member.pax_headers
                assert archive.extractfile(member).read() == path.read_bytes()
        inventory.replace(root / inventory.name)
        bundle.replace(target)
    print(f"{target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    pack()
