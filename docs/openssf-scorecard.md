# OpenSSF Scorecard — configuration & status

This repository is wired for the [OpenSSF Scorecard](https://scorecard.dev), the
automated assessment of a project's supply-chain security posture. This note
records what is configured, what each Scorecard check will report, and the one
step needed to make the assessment run live.

## Current state

The repository is **private**. Scorecard's analysis uses GitHub's GraphQL API
(e.g. `ListCommits`), which only works with the default `GITHUB_TOKEN` on **public**
repositories; on a private repo it fails with *"Resource not accessible by
integration"*. The same applies to CodeQL code scanning and to branch-protection
rules, which are free on public repositories but require GitHub Pro on private ones.

Accordingly, the `Scorecard` and `CodeQL` workflows are **gated on visibility**: they
skip cleanly while the repo is private (so the Actions tab stays green) and run
automatically the moment it is made public. Nothing else needs to change.

## Activation — going public (one step)

```sh
gh repo edit SoheilAyati/Hot-Rod-Public --visibility public --accept-visibility-change-consequences
```

That single change activates everything below:

1. The **Scorecard** workflow runs on push and weekly, publishes to the public
   OpenSSF API, and uploads results to code scanning. The README badge starts
   resolving.
2. The **CodeQL** (SAST) workflow runs and reports into code scanning.
3. **Branch protection** becomes available for free — enable it on `main`:

   ```sh
   gh api -X PUT repos/SoheilAyati/Hot-Rod-Public/branches/main/protection --input - <<'JSON'
   { "required_status_checks": { "strict": true, "contexts": ["Tests (Python 3.12)"] },
     "enforce_admins": false,
     "required_pull_request_reviews": null,
     "restrictions": null,
     "allow_force_pushes": false,
     "allow_deletions": false,
     "required_conversation_resolution": true }
   JSON
   ```

(There is also a private-repo alternative: add a read-only **classic PAT** as the
`SCORECARD_TOKEN` secret and set `repo_token: ${{ secrets.SCORECARD_TOKEN }}` on the
Scorecard step. CodeQL, the public badge, and free branch protection would still
require going public, so flipping visibility is the cleaner path.)

## What each check will report

With the configuration in this repository, once public:

| Scorecard check | Expected | Why |
| --- | --- | --- |
| Token-Permissions | **pass** | every workflow declares `permissions` minimally (`read-all` top-level, narrow per-job writes) |
| Pinned-Dependencies | **pass** | all GitHub Actions pinned by commit SHA; pip deps pinned in `tools/requirements*.txt` |
| Dangerous-Workflow | **pass** | no untrusted-checkout + script-injection patterns; `harden-runner` in the Scorecard job |
| Dependency-Update-Tool | **pass** | `.github/dependabot.yml` (pip + github-actions) |
| License | **pass** | `LICENSE` (MIT) |
| Security-Policy | **pass** | `SECURITY.md` with a private reporting channel |
| SAST | **pass** | CodeQL workflow (runs once public) |
| CI-Tests | **pass** | `ci.yml` runs the test suite + CLI smoke on every PR |
| Binary-Artifacts | **pass** | no binaries committed |
| Vulnerabilities | **pass** | small, pinned dependency surface; Dependabot alerts on |
| Maintained | **pass** | active commit history |
| Branch-Protection | partial → pass | passes once the rule above is enabled on `main` |
| Code-Review | limited | a solo student project; improves as changes land via reviewed PRs |
| Contributors | limited | inherent to a small single-author repo (wants contributors across ≥2 orgs) |
| CII-Best-Practices | n/a | requires separately registering the project at bestpractices.dev |
| Fuzzing | n/a | not applicable to these small deterministic utilities |
| Signed-Releases | n/a | no tagged releases yet; sign them if/when releases are cut |
| Packaging | n/a | not published to a package registry |

The checks the project fully controls are all green; the remainder are inherent to a
small, single-maintainer academic repository rather than configuration gaps, and are
listed here for transparency rather than hidden.

## Running Scorecard locally (optional)

You can assess the repo without waiting for the workflow using the Scorecard CLI:

```sh
export GITHUB_AUTH_TOKEN=<a token with repo read access>
scorecard --repo=github.com/SoheilAyati/Hot-Rod-Public
```

See <https://github.com/ossf/scorecard#installation> for installing the CLI.
