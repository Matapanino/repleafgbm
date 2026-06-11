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

## Before pushing ☐

- [ ] Decide the GitHub org/user and repo name; update
      `[project.urls] Homepage` in pyproject.toml (currently a placeholder:
      `github.com/repleafgbm/repleafgbm`)
- [ ] Decide author attribution in pyproject (`authors = [...]`, currently
      "RepLeafGBM contributors") and LICENSE copyright line
- [ ] `git remote add origin … && git push -u origin main`

## After the first push ☐

- [ ] Confirm CI is green on GitHub Actions; add the badge to README:
      `![CI](https://github.com/<org>/<repo>/actions/workflows/ci.yml/badge.svg)`
- [ ] Tag `v0.0.1` (the version in pyproject) once CI is green
- [ ] Branch protection on `main` (require CI) if collaborators join
- [ ] Optional: issue templates, `SECURITY.md`, PyPI publication (defer
      until the API stabilizes — the README already warns about instability)

## Working-copy location (Google Drive caveat)

The working copy currently lives under Google Drive (CloudStorage). Git and
sync clients interact badly; see the recommendations in the Phase 5 report
and prefer moving the clone to a local path with GitHub as the only sync
mechanism. At minimum, never run git operations while Drive is actively
syncing `.git/`.
