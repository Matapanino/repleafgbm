# Security Policy

RepLeafGBM is research-oriented tabular ML software, released on PyPI from
v1.0.0 onward. Model directories are loaded with `numpy.load` / JSON parsing
only — no pickle, no code execution from model files.

## Supported versions

Security fixes target the latest released version on PyPI and the latest
commit on `main`. There are no maintained older release branches.

## Reporting a vulnerability

Please report suspected vulnerabilities privately via
[GitHub security advisories](https://github.com/Matapanino/repleafgbm/security/advisories/new)
rather than opening a public issue. Include a minimal reproduction if you
can. You should receive a response within two weeks.

Issues that are *not* security-sensitive (crashes on malformed input,
incorrect results, dependency upgrade requests) are welcome as regular
GitHub issues.

## Scope notes

- Loading a model directory from an untrusted source is not a supported
  threat model: schema validation guards against corruption, not against
  adversarial inputs.
- Optional integrations (lightgbm, xgboost, torch) execute those libraries'
  native code; their security posture is inherited from those projects.
