<!-- Describe WHAT changed and WHY. Keep the diff reviewable: one topic per PR. -->

## Checklist (UW3 Style Charter — docs/developer/UW3_STYLE_CHARTER.md)

- [ ] I have read the Style Charter and followed it over the surrounding code (S2)
- [ ] No drive-by refactors, renames, or "while I was here" cleanups (S9)
- [ ] Bug fixes ship the regression test, written first, with `level_*` and `tier_*` markers (S8)
- [ ] No hedging names (`maybe_` / `try_` / `do_`) and no commented-out code (S3, S4)
- [ ] Every exception swallow states its sanctioned failure mode in a comment (S4)
- [ ] New data access uses `.array` / `mesh.X.coords` — no `with ....access(...)`, no `mesh.data` (S7)
- [ ] Parallel safety considered — np2/np4 checked where swarm/mesh/solver dispatch is touched (S11)
- [ ] No `pixi.toml` / `pixi.lock` or other dependency changes riding in an unrelated PR
- [ ] AI-assisted work carries the attribution line below

<!-- If AI-assisted, end the PR body with:
Underworld development team with AI support from [Claude Code](https://claude.com/claude-code) -->
