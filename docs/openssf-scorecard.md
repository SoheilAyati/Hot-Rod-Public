# OpenSSF Scorecard — configuration & status

This repository runs the [OpenSSF Scorecard](https://scorecard.dev), the automated
assessment of a project's supply-chain security posture. This note records the live
result, what each check reports and why, and the steps that would raise it further.

## Status: live

The repository is **public** and the Scorecard workflow runs on every push to `main`,
weekly, and on branch-protection changes, publishing to the public OpenSSF API. The
score is shown by the badge in the top-level README and at
<https://scorecard.dev/viewer/?uri=github.com/SoheilAyati/Hot-Rod-Public>.

**Current aggregate score: 6.8 / 10.** Every check the project fully controls is
maxed; the remainder are inherent to a small, new, single-maintainer repository, or
require a personal access token for Scorecard to read them.

| Check | Score | Notes |
| --- | --- | --- |
| Binary-Artifacts | **10** | no binaries committed |
| Token-Permissions | **10** | every workflow declares `permissions` minimally |
| Dangerous-Workflow | **10** | no untrusted-checkout / script-injection patterns; `harden-runner` |
| SAST | **10** | CodeQL runs on push/PR |
| Dependency-Update-Tool | **10** | Dependabot (pip + github-actions) |
| Security-Policy | **10** | `SECURITY.md` with a private reporting channel |
| Vulnerabilities | **10** | no open OSV advisories in the dependency graph |
| License | **10** | `LICENSE` (MIT), detected |
| Pinned-Dependencies | 7 | all Actions SHA-pinned and pip version-pinned; 10 needs **hash**-pinning (`--require-hashes`), which we judge not worth the maintenance/brittleness here |
| Code-Review | 0 | improves as changes land through reviewed pull requests |
| Maintained | 0 | time-based (90-day window); rises with sustained activity |
| Contributors | 0 | inherent to a single-author repo (wants contributors across ≥2 orgs) |
| CII-Best-Practices | 0 | requires separately registering at <https://www.bestpractices.dev> |
| Fuzzing | 0 | not applicable to these small deterministic utilities |
| Branch-Protection | n/a | **is enabled** on `main` (see below); Scorecard can't read classic rules with the default token — needs a PAT |
| CI-Tests | n/a | no merged pull requests yet to evaluate; populates once PRs are used |
| Packaging | n/a | not published to a package registry |
| Signed-Releases | n/a | no tagged releases yet |

## Branch protection (enabled)

`main` is protected: force-pushes and deletions are blocked, the CI checks
(`Tests (Python 3.11/3.12)`) are required, and conversation resolution is required.
Administrators are exempt so the solo maintainer can still push directly. Scorecard
reports this check as inconclusive only because the default `GITHUB_TOKEN` cannot
read classic branch-protection rules.

## Raising the score further (optional)

The controllable checks are already at 10. The realistic next steps, in rough order
of value vs effort:

1. **Use pull requests for changes.** Routing commits through PRs that run CI and get
   a review populates **Code-Review** and **CI-Tests** and lifts **Maintained** over
   time. This is the single biggest lever and needs no configuration.
2. **Add a read-only PAT for Branch-Protection.** Create a fine-grained or classic
   token with repo read access, add it as the `SCORECARD_TOKEN` secret, and set
   `repo_token: ${{ secrets.SCORECARD_TOKEN }}` on the Scorecard step. Scorecard will
   then read and score the branch-protection rules. See the
   [ossf auth docs](https://github.com/ossf/scorecard-action/blob/main/docs/authentication/fine-grained-auth-token.md).
3. **Register for the CII/OpenSSF Best Practices badge** at
   <https://www.bestpractices.dev> to score **CII-Best-Practices**.
4. **Hash-pin dependencies** (`pip-compile --generate-hashes` + `--require-hashes`) to
   take **Pinned-Dependencies** to 10 — at the cost of regenerating hashes on every
   dependency bump.
5. **Sign releases** if/when releases are cut, to score **Signed-Releases**.

## Running Scorecard locally (optional)

```sh
export GITHUB_AUTH_TOKEN=<a token with repo read access>
scorecard --repo=github.com/SoheilAyati/Hot-Rod-Public
```

See <https://github.com/ossf/scorecard#installation> for installing the CLI.
