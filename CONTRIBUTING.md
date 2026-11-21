# Contributing to multiobjective-search

Thanks for helping out. This file explains how the repository is set up and what a change
needs before it can be merged.

## Setup

1. [search-core](https://github.com/sunsided/search-core) is a regular crates.io dependency
   (`search-core = "0.3"`), and CI and Dependabot resolve it from crates.io. To develop
   against a local checkout of it, clone the planning crates side by side under one hub
   directory and put an untracked `.cargo/config.toml` in that hub (not in this
   repository):

   ```toml
   [patch.crates-io]
   search-core = { path = "search-core" }
   ```

   Every crate below the hub then builds against the sibling working tree. A patch whose
   version does not match the requirement is unused (cargo warns), and crates.io is used.
   `Cargo.lock` is not tracked (library crate), so the path entries the patch writes into
   the local lock never reach a commit. CI resolves fresh from crates.io.
2. Install Rust via [rustup](https://rustup.rs/). Development uses current stable; the MSRV
   in `Cargo.toml` (`rust-version`) is checked for every target, tests and benches included.
3. Install [Task](https://taskfile.dev/installation/). All commands below go through it. Run
   `task` to list them.
4. Optional tools: the tasks tell you when one is missing.
   - [cargo-nextest](https://nexte.st/) for faster tests;
   - [cargo-deny](https://github.com/EmbarkStudios/cargo-deny) and
     [cargo-hack](https://github.com/taiki-e/cargo-hack);
   - [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov) for coverage;
   - [typos](https://github.com/crate-ci/typos);
   - [mermaid-cli](https://github.com/mermaid-js/mermaid-cli) when changing diagrams;
   - [actionlint](https://github.com/rhysd/actionlint) and [zizmor](https://docs.zizmor.sh/);
   - nightly Rust plus [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) for fuzzing.
5. Install the git hooks with [prek](https://prek.j178.dev/) (`prek install`), or with
   `pre-commit install`. Fast checks run on commit; clippy and tests run on push.

## Everyday commands

| Command | What it does |
|---|---|
| `task test` | Tests (nextest if installed) and doctests |
| `task lint` | clippy (pedantic), separator ban, README link check, typos, workflow lints |
| `task fmt` | Format the code |
| `task doc` | Build the docs, private items included, with warnings as errors |
| `task ci` | Everything CI runs except the coverage upload and the semver check (`task coverage`, `task semver`). Run it before you open a PR |
| `task bench` | Benchmarks |
| `task test:prop` | Property tests with 10000 cases (`PROPTEST_CASES`) |
| `task fuzz:smoke` | Run every fuzz target briefly (needs nightly; `MAX_TIME=60` for longer) |
| `task clean` | Remove build output (fuzz corpora and crash artifacts are kept) |

## Code style

- Every item has a doc comment, private items included.
- No `mod.rs`: use `foo.rs` plus `foo/`.
- No `// ----` separator comments. Split the file into modules instead.
- Keep functions small, and extract helpers with descriptive names.
- `clippy::pedantic` must pass. If a lint is wrong in one place, use
  `#[expect(lint, reason = "...")]` on that item. Never `#[allow]`.
- Unsafe code is forbidden unless a crate explicitly documents why it needs it.
- README and doc links use absolute URLs (crates.io renders READMEs outside GitHub).
  Mermaid diagrams are welcome.

## Branches

This repository uses **GitHub Flow**:

- `main` is always releasable.
- Work happens on `feature/<topic>` branches. Bug fixes are feature branches too.
- Open a PR into `main`.

No other prefixes (`fix/`, `chore/`, `deps/`, ...). `task lint:branch`, the pre-push hook and
CI check the branch name. Bot branches (`release/*` from release-plz, `dependabot/*`) are the
only exceptions. Copilot's `copilot/*` branches are renamed to `feature/*` before anyone
works on them.

## Commits and pull requests

- Commit subjects use the **imperative mood**, like Git itself: "Add parser for pair lists",
  "Fix overflow in tile index". Do not write "Added ...", "Adds ..." or `feat: ...`.
- Keep the subject to 72 characters or fewer, capitalized, with no trailing period. Then add
  a blank line and a body that explains *why*. The `commit-msg` hook checks the subject.
- No co-author trailers, claude.ai/claude.com links or AI attribution in commit messages. The
  `commit-msg` and `pre-push` hooks reject them (`scripts/check-no-ai-attribution.sh`).
- The changelog groups commits by their leading verb (Add, Fix, Remove, Deprecate, ...).
  Write a `BREAKING CHANGE:` line in the body for incompatible changes.
- PRs are merged with a merge commit, or squashed using the PR title and description. Write
  both with care. Head branches are deleted after the merge.
- Keep PRs focused. CI must be green.
- Do not bump versions or edit `CHANGELOG.md` by hand. Releases are automated (see below).

## Releases

Versions follow [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html), as Cargo
implements it (in `0.x`, a minor bump is breaking).

Release automation is dormant until the repository variable `RELEASE_APP_CLIENT_ID` (and the
secret `RELEASE_APP_PRIVATE_KEY`) of the release GitHub App is set: until then the release-plz
jobs are skipped. Setting the variable activates releases.

Releases are handled by [release-plz](https://release-plz.dev/). Every merge to
`main` updates a release PR. Merging that PR tags the release and creates the
GitHub Releases. The `vX.Y.Z` tag then triggers `publish-crates-io.yml`, which
publishes the crate to crates.io via trusted publishing from the protected `crates-io`
environment.

## License

By contributing, you agree that your contributions are licensed under MIT OR Apache-2.0, the same
terms as the project. See the License section of the README.
