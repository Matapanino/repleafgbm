# GitHub Publication Checklist

Status of each item as of v1.0.2 (Phase 28, 2026-06-15). Items marked ☐ need a
human decision or a GitHub-side action that cannot be done from this repo alone.

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
- [x] Branch protection on `main` (require CI)
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

- [x] **PyPI trusted publisher**: `repleafgbm`, owner `Matapanino`, repo
      `repleafgbm`, workflow `publish.yml` — configured on pypi.org.
- [x] **Push the release tag** `v1.0.0` → `publish.yml` → PyPI (v1.0.1
      followed; see `CHANGELOG.md`)
- [x] Branch protection on `main` (require CI)
- [x] GitHub Pages for the built API docs

## Working-copy location

The canonical working copy lives at a local path (`~/dev/repleafgbm`) with
GitHub as the only sync mechanism; the earlier Google Drive (CloudStorage)
location was abandoned because git and the sync client interacted badly.
