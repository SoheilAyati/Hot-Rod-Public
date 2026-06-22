# Security policy

This repository publishes documentation (model cards) and a small set of
offline-first Python utilities for an academic forecasting competition. It is not
production infrastructure, but we take dependency hygiene and reproducibility
seriously and welcome reports.

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue for a
vulnerability.

- Preferred: open a private report through GitHub's
  [**Report a vulnerability**](https://github.com/SoheilAyati/Hot-Rod-Public/security/advisories/new)
  (Security → Advisories). This keeps the details confidential until a fix is ready.
- Alternative: email the maintainer at **soheilayati80@gmail.com** with a clear
  description and, if possible, a minimal reproduction.

We aim to acknowledge a report within **5 working days** and to agree on a
disclosure timeline with the reporter. Because this is a student project, response
times may be longer during examination periods; thank you for your patience.

## Scope

In scope:

- The Python tools under [`tools/`](tools/) (input handling, CSV parsing, the
  evaluation code).
- The CI / GitHub Actions workflows under [`.github/`](.github/).
- Any vulnerable or malicious dependency pulled in by `tools/requirements*.txt`.

Out of scope:

- The accuracy or correctness of a *forecast* (this is a competition, not a
  guarantee — see the model-card disclaimers).
- Third-party services the tools talk to (e.g. the ENTSO-E Transparency Platform).
- Social-engineering or physical attacks.

## Supported versions

The repository is rolling; only the latest commit on the default branch (`main`) is
supported. There are no long-term support branches.

## Our posture

The tools deliberately keep a **small dependency surface** (numpy, pandas, and an
optional ENTSO-E client) to minimise the CVE/attack surface, pin their dependencies,
and never execute untrusted input. GitHub Actions are pinned by commit SHA,
`Dependabot` watches the dependencies and actions, and the OpenSSF Scorecard workflow
tracks the repository's supply-chain hygiene over time.
