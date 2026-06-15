# GitHub Publication Checklist

Status of each item as of Phase 5 (2026-06-11). Items marked ☐ need a human
decision or a GitHub-side action that cannot be done from this repo alone.

## Ready ✅

- [x] MIT LICENSE, README with honest status/warnings, CONTRIBUTING.md
- [x] Test suite (`python -m pytest tests/ -q`, 100+ tests) and
      `scripts/check.sh` one-shot gate
- [x] CI workflow (`.github/workflows/ci.yml`: lint + tests + examples on
      Python 3.10/3.12) — activates automatically on push
- [x] Packaging (`pyproject.toml`, src layout, optional extras:
      `external`, `bench`, `torch`, `dev`)
- [x] No secrets/credentials in the tree; synthetic data only
- [x] Docs: design / math / roadmap (implemented vs planned kept honest) /
      serialization / backend strategy / categorical policy / ADRs /
      experiment reports

## Before pushing ✅ (done 2026-06-11, Phase 9)

- [x] Repo: `Matapanino/repleafgbm` (public); pyproject Homepage/Repository
      URLs updated
- [x] Author attribution: Masaya Kawamata in pyproject and LICENSE
- [x] Pushed via `gh repo create … --source . --push`

## After the first push (status as of 2026-06-11)

- [x] CI green on GitHub Actions (first run caught a real pandas>=3
      string-dtype bug in categorical auto-detection — fixed); badge added
      to README
- [x] Tagged `v0.0.1`
- [ ] Branch protection on `main` (require CI) if collaborators join
- [x] Issue templates (`.github/ISSUE_TEMPLATE/`) and `SECURITY.md`
      (private reporting via GitHub security advisories) — added 2026-06-12,
      Phase 20
- [x] Move the working copy out of Google Drive to a local path
      (`~/dev/repleafgbm`); GitHub is now the canonical remote

## v1.0.0 release (Phase 27, 2026-06-15)

API stabilized in Phase 24 (ADR 0003), so the PyPI deferral is lifted.

Done in this repo:

- [x] Version bumped to `1.0.0` (`pyproject.toml`, `__init__.py`); classifier
      `Development Status :: 5 - Production/Stable`
- [x] `python -m build` + `twine check dist/*` pass locally (pure-Python
      universal wheel; the Rust `native/` extension is built separately and is
      not part of the PyPI distribution for 1.0)
- [x] `.github/workflows/publish.yml` — OIDC trusted publishing on `v*` tags
      (manual dispatch → TestPyPI dry run)
- [x] `CHANGELOG.md`

Manual one-time actions (PyPI / GitHub side — cannot be done from the repo):

- [ ] **PyPI trusted publisher**: pypi.org/manage/account/publishing — add
      publisher `repleafgbm`, owner `Matapanino`, repo `repleafgbm`, workflow
      `publish.yml`, environment blank. (Optional: repeat on test.pypi.org and
      run the workflow manually for a TestPyPI dry run.)
- [ ] **Push the release tag** `v1.0.0` → triggers `publish.yml` → PyPI
- [ ] Branch protection on `main` (require CI) if collaborators join

## Working-copy location (Google Drive caveat)

The working copy currently lives under Google Drive (CloudStorage). Git and
sync clients interact badly; see the recommendations in the Phase 5 report
and prefer moving the clone to a local path with GitHub as the only sync
mechanism. At minimum, never run git operations while Drive is actively
syncing `.git/`.
