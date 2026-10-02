# Release procedure

MantiShrimp releases are published from GitHub. A published GitHub Release
triggers the PyPI Trusted Publishing workflow and, once the repository is
enabled in Zenodo, its software archive.

## Updating the existing release

Version [v0.1.0](https://github.com/sthsci/MantiShrimp/releases/tag/v0.1.0)
is already published on GitHub and PyPI, with a
[Zenodo archive](https://zenodo.org/records/21908641). Check that the repository
is still enabled in [Zenodo's GitHub integration](https://zenodo.org/account/settings/github/)
before publishing the next release; there is no need to repeat the initial
account setup if it is still configured.

The additive bound-adhesion change is version **0.2.0**:
it changes the default scientific model and its trajectories. Do not reuse the v0.1.0 tag
or its version DOI `10.5281/zenodo.21908641` for new results. The concept DOI
`10.5281/zenodo.21908640` identifies the version family; cite the new version's
DOI when reporting simulations run with the changed mechanics.

Commit only the intended files, because the working tree may contain other
work. For this mechanics change:

```bash
git switch -c codex/additive-bound-adhesion
git add src/mantishrimp/simulation.py script/simulator/assistant_function.py \
  tests/test_simulation.py docs/model.md docs/releasing.md CHANGELOG.md \
  pyproject.toml src/mantishrimp/__init__.py CITATION.cff
git add figures/abm_schematic_drafts/plot_force_and_binding.py \
  figures/abm_schematic_drafts/force_and_binding_rules_additive.png \
  figures/abm_schematic_drafts/force_and_binding_rules_additive.svg \
  figures/abm_schematic_drafts/additive_caption.md
git diff --cached
git commit -m "Add bound adhesion to overlap repulsion"
git push -u origin codex/additive-bound-adhesion
```

Open a pull request, wait for CI, and merge it before publishing the release.
Stage final figure files explicitly; do not stage an entire drafts or paper
directory automatically. A branch push does not publish
to Zenodo or PyPI. **Publishing the GitHub Release does trigger the PyPI
workflow and, when enabled, Zenodo archiving.**

## One-time account setup

1. Create and verify a PyPI account with two-factor authentication.
2. In the GitHub repository settings, create an environment named `pypi` and
   require a maintainer's approval for deployment.
3. In the PyPI account's **Publishing** page, add a pending GitHub publisher:

   - project: `mantishrimp`
   - owner: `sthsci`
   - repository: `MantiShrimp`
   - workflow: `release.yml`
   - environment: `pypi`

4. Sign into Zenodo with GitHub, open the GitHub integration, click **Sync
   now**, and enable `sthsci/MantiShrimp`.

Complete these steps before publishing the first GitHub Release. A pending
PyPI publisher does not reserve the project name until the first successful
upload.

## Release checklist

1. Update the version in `pyproject.toml`, `src/mantishrimp/__init__.py`, and
   `CITATION.cff`, including its `date-released` field.
2. Move the relevant entries from `CHANGELOG.md` under the new version and set
   its release date.
3. Confirm author, ORCID, affiliation, and contribution metadata.
4. Run:

   ```bash
   python -m pip install -e '.[all,test]'
   python -m pytest
   python .github/scripts/check_release_version.py
   python -m build
   python -m twine check --strict dist/*
   ```

5. Push the release-preparation commit and wait for the full CI matrix.
6. On GitHub, create a release from `main` with a new tag matching the package
   version, for example `v0.2.0`. Use the changelog entry as release notes.
7. Publish the GitHub Release. Do not create the tag separately: GitHub creates
   it from the selected release target.
8. Approve the `pypi` deployment when the release workflow requests it.
9. Verify the PyPI page and a clean installation of the base and inference
   extras.
10. Wait for Zenodo to finish archiving and check the creator metadata, version
    DOI, and access settings. Verify that the source archive can be downloaded
    publicly if the release is intended to be open. Add the resulting DOI
    badge to the README in a follow-up commit.

PyPI release files and Zenodo software versions are persistent. If code or
package metadata must change after publication, increment the package version
instead of trying to replace an existing version.

Official instructions: [GitHub releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository),
[Zenodo repository integration](https://help.zenodo.org/docs/github/enable-repository/),
[Zenodo archiving and troubleshooting](https://help.zenodo.org/docs/github/archive-software/github-upload/),
and [DOI versioning](https://zenodo.org/help/versioning).
