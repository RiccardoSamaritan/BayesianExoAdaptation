# Refactor notes

Log of the structural/didactic pass over this repository, and a list of things
that look like candidates for removal but were deliberately **not** deleted
(per the conservation rule) — reviewed and confirmed before doing anything
about them.

## What was changed

- Added `src/__init__.py` (empty package docstring listing every module) so
  `src` is an explicit package rather than relying on implicit namespace-package
  behaviour. Purely additive: the `sys.path.insert(...); from src.X import ...`
  pattern every notebook already uses is untouched and still works.
- Moved `a.md` to `docs/project_overview_it.md` (`git mv`, history preserved)
  and added a short header clarifying that it describes the project's original
  sEMG/BASAN framing, pre-dating the pivot to UCI HAR — content below the
  header is otherwise byte-for-byte the original text.
- Added a "Repository layout" section to `README.md` and corrected its
  `## Status` line, which still read "No code implemented yet" despite
  `TODO.md` showing §2-§10 complete. `TODO.md` itself was left untouched, per
  instructions.
- Removed OS/build clutter with no content value: tracked `.DS_Store` files
  (`.DS_Store`, `data/.DS_Store`, `data/test/.DS_Store`, `notebooks/.DS_Store`)
  and the untracked `src/__pycache__/` bytecode cache. Added `.DS_Store` to
  `.gitignore` so it doesn't get re-committed.
- Lightly rewrote the intro and closing "Summary" markdown cells of
  `notebooks/loader.ipynb` (cells 0 and 20 only — every other cell, all code,
  and all outputs are untouched). This was the one notebook that was
  noticeably more verbose/checklist-style ("✓ Import successful!",
  exclamation-heavy prose) than the rest of the suite. While doing this I
  found the original closing summary overstated its own result ("Target
  subjects show measurable distribution shift") when the notebook's own
  executed output two cells earlier prints a 1.20x ratio against its own 2x
  threshold and literally logs "⚠ Target shifts are comparable to source
  baseline - minimal shift detected." The rewritten summary now quotes the
  actual numbers (14.54 vs. 12.15, ratio 1.2x).

No AI-tool references (ChatGPT/Claude/Copilot/Anthropic/OpenAI/etc.) were
found anywhere in the repository (`src/`, `notebooks/`, docs, `.gitignore`) —
nothing needed removing there.

## Resolved (approved and executed)

### 1. Superseded draft notebook + its unused module — deleted

Verified all four conditions before deleting, per the approved plan:

1. `grep -rn "uncertainty_utils" .` — the only import was in
   `notebooks/har_uncertainty_decomp.ipynb`; the sole other hit was a
   docstring mention in `src/__init__.py`'s module list, since fixed (removed
   the `uncertainty_utils` line from that docstring, as the module no longer
   exists).
2. `data/har_6class_model.pt` and `data/har_6class_laplace.npz` do not exist
   on disk, and no other notebook/script references either path.
3. `TODO.md` §7 names `har_uncertainty_decomposition.ipynb` once and never
   names `har_uncertainty_decomp.ipynb`.
   Correction to my own original note here (now superseded): I had written
   "cites it by name three times" — rechecked with `grep -c`, it's cited
   once. The substance (long name present, short name absent) still holds;
   only the count was wrong.
4. `har_uncertainty_decomposition.ipynb` imports `normalize_epistemic` from
   `src/bayesian`, not from `src/uncertainty_utils`.

All four passed, so both files were deleted in a single, isolated commit
(`1bc714d`, "Remove superseded §7 draft notebook and its unused
uncertainty_utils module (resolves normalize_epistemic name collision)"),
touching only those two paths.

### 2. `normalize_epistemic` name collision — resolved by deleting #1

`src/uncertainty_utils.py::normalize_epistemic` (the quantile-ratio version)
is gone. The surviving `src/bayesian.py::normalize_epistemic` (`mode="log_k"`
/ `"source_quantile"`) is unchanged and still used by `src/im_adapt.py` and
`notebooks/har_uncertainty_decomposition.ipynb`.

### 3. Audit findings 1-4 — fixed (finding 5 deliberately left alone, per instruction)

All four content/numeric issues surfaced by the markdown-vs-output audit
(see the audit section below for how each was found) were corrected directly
in the affected notebooks. Only markdown cell text was touched in each case;
code cells and existing outputs elsewhere in the same notebooks are
untouched.

- **`notebooks/har_uncertainty_decomposition.ipynb`, cells 8 and 24**
  ("Finding 1"): "60x" corrected to "8.9x" (0.0515 / 0.0058 = 8.88), matching
  both the notebook's own code-cell output and `TODO.md`'s citation of the
  same result.
- **`notebooks/har_calibration.ipynb`, cell 17** ("Finding 2"): rewritten to
  state accurately that epistemic AUROC underperforms MAP/total entropy only
  on the moderately-shifted, still-fairly-accurate subjects 7/8, while it is
  actually the *best* of the three detectors on the two lowest-accuracy
  subjects (20, 19) — the reverse of what the original text claimed for
  "the hardest" subjects. Exact AUROC values from the notebook's own table
  are now quoted in the text.
- **`notebooks/har_calibration.ipynb`, cells 14 and 25** ("Finding 3"):
  reworded to list the actual 7 subjects where Laplace improves ECE/NLL
  (7, 8, 11, 19, 20, 21, 22 — not just the 5 harder ones originally named),
  and to note that two of them (11, 21) are easy/high-accuracy subjects with
  only a small improvement, rather than claiming the improvement is
  "consistently" on the harder ones.
- **`notebooks/synthetic_track.ipynb`** ("Finding 4"): the notebook had never
  been executed (`execution_count: None`, empty outputs, on every code cell).
  Executed it in place (`jupyter nbconvert --execute --inplace`, CPU, ~under
  a minute) — all 7 code cells ran with zero errors, including the two cells
  whose `assert` statements enforce the notebook's own semantic-check and
  Spearman-correlation claims. Notable results now visible in the saved
  output: the "mild" shift regime keeps 100% accuracy under both plain and
  uncertainty-guided adaptation (the intended negative control); the
  "strong" shift regime goes from 68.2% (blue-only 6.0%) before adaptation,
  to 68.4% (7.3%) after conventional IM (the paper's expected near-failure),
  to 97.1% (98.7%) after uncertainty-guided IM — a clean, large rescue effect
  matching the notebook's narrative; the semantic check and the shift-vs-
  epistemic Spearman correlation (ρ=0.927, p=1.12e-04) both pass as asserted.

**Finding 5 (the local absolute path leaked into `har_openset.ipynb`'s saved
output) was deliberately left untouched**, per instruction.

## Flagged, not touched — needs your call

### 4. `data/` mixes raw datasets with generated inference-cache artifacts

`data/har_bald_artifact.npz` and `data/har_bald_artifact_openset.npz` are
pipeline outputs (written by `har_mc_convergence.ipynb` and
`har_openset.ipynb`), sitting next to the raw UCI HAR / HAPT files they were
computed from. Separating raw data from generated caches (e.g. a
`data/cache/` subfolder) is more conventional, but every notebook hardcodes
`"../data"` (8+ call sites across 6 notebooks) as the single data root passed
to `load_har`/`load_hapt`, and the two artifact-producing/consuming notebooks
hardcode the `.npz` paths directly. Moving these would mean editing functional
code cells in multiple notebooks for a purely cosmetic gain, which the
conservation rule for this pass explicitly discourages — left as-is,
flagged here in case you want to do it deliberately later.

## Audit: markdown claims vs. actual executed outputs (how findings 1-5 were found)

Every notebook's interpretive markdown (prose paragraphs and closing
"Summary" cells, not code) was checked against that same notebook's own
saved `stdout`/output cells — i.e. does the notebook's own prose accurately
describe the numbers its own code produced? `loader.ipynb` was excluded
(corrected in an earlier pass). This section is a historical record of how
findings 1-4 (now fixed above) and finding 5 (left alone) were identified;
nothing here still needs action except finding 5, which was explicitly
excluded from this round.

### Finding 1 (fixed) — numeric error: "60x" should be "8.9x"

`har_uncertainty_decomposition.ipynb` cells 8/24 claimed a 60x epistemic
elevation for the hardest target; the notebook's own code cell 10 computes
and prints 8.9x (0.0515 / 0.0058), matching `TODO.md`'s citation of the same
result. Corrected in both cells.

### Finding 2 (fixed) — interpretive error: epistemic AUROC on the truly hardest subjects

`har_calibration.ipynb` cell 17 claimed total/MAP entropy do "about as well
or better" than epistemic on "the hardest, most-shifted subjects." The
printed AUROC table (cell 16) shows the opposite on the two lowest-accuracy
subjects (20: epistemic 0.811 > total 0.770 > MAP-ent 0.733; 19: epistemic
0.893 > total 0.857 > MAP-ent 0.802) — epistemic is only weaker on the
moderately-shifted, still-accurate subjects 7/8. Rewritten to state this
precisely. (Near-identical phrasing exists in `TODO.md` §8, left untouched
per instructions — a pre-existing narrative choice, not unique to the
notebook.)

### Finding 3 (fixed) — minor imprecision: "consistently the harder, higher-shift ones"

`har_calibration.ipynb` cells 14/25 named only 5 of the 7 subjects where
Laplace actually improves ECE/NLL, omitting 11 and 21 (both easy/high-
accuracy), and used "consistently" to describe a pattern that holds for 5 of
7. Reworded to list all 7 and qualify the omission. (Same phrasing also
exists in `TODO.md` §8, left untouched per instructions.)

### Finding 4 (fixed) — gap: `synthetic_track.ipynb` had never been executed

Every code cell had `execution_count: None` and no saved outputs. Executed
in place; all cells ran cleanly including both `assert`-guarded checks. See
the "Resolved" section above for the resulting numbers.

### Finding 5 (left untouched, per instruction) — local file path in a saved output

`har_openset.ipynb` code cell 13's saved output contains an absolute path
from the original author's machine
(`/Users/riccardo/Desktop/PML/BayesianExoAdaptation/data/har_bald_artifact_openset.npz`).
Low sensitivity — the GitHub remote (`git log`) already publicly attributes
this repo to the same name — left alone as explicitly requested.

### Everything else checked out

`test_hessian_binary.ipynb`, `test_laplace_vs_mcmc.ipynb` (including its
correlation numbers as separately quoted in `har_uncertainty_decomposition.ipynb`'s
markdown), `mnist_check.ipynb` (including every ratio and rotation-sweep
number, which also match `TODO.md` §3b's citations exactly), `har_source_training.ipynb`,
`har_mc_convergence.ipynb`, and `har_adaptation_ablation.ipynb` (every
tercile, collapse-rate, and Δaccuracy claim checked against its own printed
tables) all had their narrative claims match their own recorded outputs
precisely, including numbers quoted to 3-4 significant figures. One claim in
`har_adaptation_ablation.ipynb` (subject 19, arm (f), zero predictions on
WALKING_DOWNSTAIRS, class counts "52/40/39") could not be verified either way
— it's a per-sample-per-class breakdown that was never printed to the
notebook's saved output, so it's neither confirmed nor contradicted by what's
on disk.
