# Security Policy

RepLeafGBM is experimental research software (pre-1.0, no PyPI release).
It is not intended for production use, and model directories are loaded
with `numpy.load` / JSON parsing only — no pickle, no code execution from
model files.

## Supported versions

Only the latest commit on `main` is supported. There are no maintained
release branches.

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
