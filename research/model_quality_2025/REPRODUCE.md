# Reproduce the recorded comparison

Run from the repository root with the project dependencies installed. The
frozen input tables are included in `data.pkl`; load only this trusted snapshot,
since Python pickle files can execute code. Its SHA-256 is recorded in
`manifest.json`. The tables contain public historical returns and macro data,
not account data or credentials.

Do not run `--prepare` to reproduce this particular result: it downloads newer
vintages and selects the current HEAD as the baseline. Instead reconstruct the
recorded baseline from Git history:

```sh
mkdir -p research/model_quality_2025/baseline_package
git archive 3d5cbc7 src | tar -x -C research/model_quality_2025/baseline_package
.venv/bin/python scripts/compare_model_quality.py --variant baseline --package research/model_quality_2025/baseline_package/src
.venv/bin/python scripts/compare_model_quality.py --variant updated --package src
.venv/bin/python scripts/compare_model_quality.py --report
```

These commands overwrite the recorded result files. Preserve them elsewhere
first if you want to compare the rerun against the original. Use the source
version accompanying this report; subsequent model changes alter the updated
result. Platform/library differences can also change numerical results.

The duplicate baseline package, generated native libraries, and partial
checkpoints are deliberately excluded from version control. The complete
results, report, input snapshot, and provenance manifest are included.
