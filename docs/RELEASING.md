# Release preparation

The 0.1.0 implementation is experimental. A wheel and source distribution can be built
without fetching any benchmark data. PyPI publication has not been authorized for this
session and has not been performed.

## Checks available offline

1. Run `pytest -q` and `ruff check src tests`.
2. Build with `python -m build` and check with `twine check dist/*`.
3. Install the wheel into a fresh environment, then run `banana doctor` and `banana smoke`.
4. Inspect the distribution for accidental credentials, task data, or development artifacts.

The test suite uses original synthetic fixtures only. It covers native execution, patch
replay, private-case encoding, timeout cleanup, budgets, fingerprints, resume, and rootfs
command contracts. The optional providers, original graders, and image executions require
integration validation before claiming benchmark compatibility beyond these contracts.

## Validation to perform only in an explicitly authorized environment

- Run one reference/oracle patch and one empty patch against each intended SWE-bench
  Verified repository family using the pinned 4.1.0 grader and published instance image.
- Validate current Pro V2, DeepSWE separate-verifier collection, and Terminal-Bench 2.1
  with their original images and verifier files. Record resolved source commits and images.
- Cross-check LiveCodeBench stdout, functional, float, timeout, and runtime-error cases
  against the selected upstream evaluator snapshot; record generation protocol differences.
- Run rootfs P1/P2 on Colab, Molab, and a nested unprivileged container. Confirm ptrace
  availability, filesystem mappings, cleanup, image architecture, disk space, and timing.
- Exercise real provider API responses, usage accounting, and context/request limits.

These checks were **not** run in this repository. No benchmark data or images are bundled
in the package. Real benchmark scores must not be reported as official leaderboard scores
for this host runtime. No runtime can guarantee every task works on an arbitrary notebook.

## Publication, later

Check that `banana-harness` is available on PyPI, select the release version, and complete
the desired integration checks. Package-name availability has not been checked here.
Build in a clean checkout and inspect wheel contents. When the owner chooses to publish,
use PyPI trusted publishing or run `python -m twine upload dist/*` with their credentials.
Do not put PyPI tokens in notebooks or commit them to the repository.

