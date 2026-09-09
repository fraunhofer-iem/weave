# ML4SRM Replication Package

This folder contains the resources needed to reproduce the empirical and explainability results for the ML4SRM approaches. Included are the
datasets, pipeline configuration, SLURM job scripts, SHAP outputs, run logs, and the analysis scripts.

The headline comparison is **`published` vs `weave`**. Each case study explains two classifiers
on the same data, the model the tool shipped with and the model ML-Plan selected in stage 1.

It is possible to recompute or reproduce the results by following the necessary steps. *Recompute* regenerates the figures and tables from the committed results.

```bash
analysis/reproduce.sh
```

*Reproduce* re-runs WEKA and the SHAP explainer from the ARFFs to produce those outputs in the
first place. See
[Reproducing a run](#reproducing-a-run). 


## ML4SRM Case Studies

Seven case studies, one per detection target, across the three ML4SRM approaches. SSCM is one
three-class model, Dev-Assist one multi-label model over five labels, and SWAN five separate
binary models, one per label. Each carries its own feature set, so the counts below vary by case
study and not just by approach.

| Directory | Toolkit | Target | Features | Train rows | Test (GXA) rows |
|---|---|---|---|---|-----------------|
| `sscm/` | weka | `tag {None,Target,Input}` (3-class) | 103 | 9021 | 250             |
| `dev-assist/` | meka | 5 labels: `CWE79`, `source`, `sanitizer`, `CWE89`, `sink` | 121 | 9229 | 250             |
| `swan/source/` | weka | `source {0,1}` | 68 | 9229 | 250             |
| `swan/sink/` | weka | `sink {0,1}` | 84 | 9229 | 250             |
| `swan/sanitizer/` | weka | `sanitizer {0,1}` | 48 | 9229 | -               |
| `swan/cwe79/` | weka | `CWE79 {0,1}` | 26 | 9229 | -               |
| `swan/cwe89/` | weka | `CWE89 {0,1}` | 22 | 9229 | 250             |

`dev-assist` is the multi-label case. MEKA moves its 5 labels to attribute indices `0..4`,
which is the order the `label_<n>` suffixes refer to. So `label_0` = `CWE79`, `label_1` = `source`,
`label_2` = `sanitizer`, `label_3` = `CWE89`, `label_4` = `sink`.

## Datasets

Each case study ships two ARFFs with an identical attribute schema:

- **`<name>-dataset.arff`**: the training set. Used to fit the model and as
  the population for the *global* explanations.
- **`<name>-gxa-dataset.arff`**: the held-out set, drawn from the Gene
  Expression Atlas (`uk.ac.ebi.gxa`) codebase. Used for the *local*
  explanations, so every waterfall plot corresponds to a real Java method.
  These ARFFs are extracted features, not labels. The subject program, the two-labeller
  annotation, the consensus rule and the seeded 250-method sample are documented in
  [SRM-Dataset](https://github.com/secure-software-engineering/srm-dataset), under
  `datasets/java/sources/gxa/`.
- **`<name>-gxa-methods[.txt]`**: the method signatures for that held-out set,
  one per line, in ARFF row order. This is what maps a SHAP row back to a
  method. Present for `sscm`, `dev-assist`, `swan/cwe89`, `swan/sink` and
  `swan/source`.
- **`feature-mapping`** maps the short feature names used in the raw data to the descriptive
  names that appear in the ARFFs and in the SHAP output. A reference table, not read by any
  script.
- **`analysis/inputs/model-selection-dump.sql`** is a mysqldump of the jaicore-experimenter database that stage 2 writes to,
  holding the repeated-cross-validation log behind the RQ1 numbers and Table 3, one row per seed
  and configuration. `analysis/cvlog.py` is the reader, and `python3 analysis/cvlog.py` prints
  the 17 configurations it finds.

## Replication Package Layout

```
evaluation/ml4srm/
├── README.md                    this file
├── requirements.txt             pinned Python deps for the SHAP subprocess
├── setup-venvs.sh               builds one venv per case study (login node)
├── feature-mapping              CSV: short → descriptive feature names
├── <case>.sh                    stage-3 SHAP job script, one per case study
├── slurm-logs/                  superseded and failed runs
│
├── analysis/                    turns the SHAP outputs into the paper's results
│   ├── reproduce.sh             regenerate every figure and table
│   ├── make_shap_figs.py        RQ2/RQ3 figures, and the direction-shift analysis
│   ├── make_perf_tables.py      RQ1 per-class performance from the cross-validation log
│   ├── make_tables.py           the five tables the paper reports, as LaTeX
│   ├── cvlog.py                 reads the experimenter database out of the SQL dump
│   ├── inputs/
│   │   └── model-selection-dump.sql   the one input that is not a SHAP output
│   └── output/                  generated, not committed
│       ├── rq1/, rq2/           figures
│       ├── tables/              table1_*.tex through table5_*.tex
│       └── logs/                one log per generator
│
├── sscm/                        weka, 3-class
│   ├── sscm-dataset.arff        train
│   ├── sscm-gxa-dataset.arff    test / local explanations
│   ├── sscm-gxa-methods.txt     row → method signature
│   ├── gxa.csv                  the GXA methods as a readable table
│   ├── sscm.properties          WEAVE pipeline config
│   ├── experiments.cnf          jaicore experiment definition (stage 2)
│   ├── db.properties            MySQL connection (stage 2)
│   ├── model-selection/         stage-1 job script and its SLURM log
│   └── explanations/
│       ├── published/           SHAP output for the model the tool shipped with
│       └── weave/               SHAP output for the ML-Plan model
│
├── dev-assist/                  meka, 5 labels; same shape as sscm, except
│   └── config/                  experiments.cnf and db.properties live here
│
└── swan/                        weka, five binary labels
    ├── db.properties            shared by all five labels
    ├── <label>.properties       one per label
    └── <label>/                 datasets, experiments.cnf, model-selection/, explanations/
```

## Reproducing a Run

Set `WEAVE_HOME` to your checkout. Every path in the job scripts and the
`.properties` files is derived from it. The `.properties` files use
`${WEAVE_HOME}` and `${MODEL_TAG:-published}` placeholders, which WEAVE expands from
the environment at startup.

**1. Build the jar** (from the repository root):

```bash
mvn -q clean package        # → target/cli-1.0-jar-with-dependencies.jar
```

**2. Build the Python environments, on a login node, once:**

```bash
export WEAVE_HOME=/path/to/weave
bash evaluation/ml4srm/setup-venvs.sh
```

Compute nodes do not reliably have outbound HTTPS. A job that installs there fails partway and
leaves a broken venv behind. `setup-venvs.sh` writes a `.weave-deps-ok` sentinel that the job
scripts check. Once the environments exist, the jobs never touch the network.

The pins in `requirements.txt` are load-bearing. `shap` pulls in `numba`, which enforces a NumPy
upper bound at *import* time. Unpinned installs therefore fail with an `ImportError`, not a
resolver conflict.

### Stage 1: Model Selection

**Re-running model selection** instead of reusing the committed descriptors is what
`<case>/model-selection/*.sh` is for, the same invocation without `-X`.

Each `model-selection/` also holds the SLURM log of the run that selected that case study's
`weave` model. That log is the record of the search: its `ML-Plan Options` line carries the seed
and the timeouts, and its `model selected` line carries the descriptor the stage-3 script reuses.

| Case study | job | `mlplan.seed` |
|---|---|---|
| `sscm` | 29700613 | 12345678 |
| `dev-assist` | 29684002 | 5570 |
| `swan/source` | 29680818 | 557 |
| `swan/sink` | 29642295 | 557 |
| `swan/sanitizer` | 29770748 | 3012026 |
| `swan/cwe79` | 29683998 | 54512570 |
| `swan/cwe89` | 29783793 | 54613374 |

Each `.properties` carries the ML-Plan settings its own run used:
`mlplan.timeout = 480`, `mlplan.timeout.node = 30`, `mlplan.cpu = 64`, the seed above, and
`mlplan.timeout.candidate = 5` for `sscm` and `dev-assist`, `10` for the five SWAN labels. Selection is still wall-clock bounded, so the recorded seed makes a re-run comparable rather than
bit-for-bit identical, and it may return a different pipeline.


The table below summarizes the published and WEAVE-selected models. 
Each row is the pair compared for that case study, written in abbreviated form. `⊃` means the
classifier on the left wraps the one on the right, so `Bagging` ⊃ `RandomForest` is bagged random
forests. Six of the seven published models are a plain SVM, which is what SWAN and SSCM shipped.

| Case study | `published` | `weave` (ML-Plan selection) |
|---|---|---|
| sscm | `SMO` | `Bagging` ⊃ `RandomForest` |
| dev-assist | `EnsembleML` ⊃ `PS` ⊃ `LMT` | `EnsembleML` ⊃ `PS` ⊃ `J48` |
| swan/source | `SMO` | `RandomForest` |
| swan/sink | `SMO` | `LogitBoost` ⊃ `RandomForest` |
| swan/sanitizer | `SMO` | `Bagging` ⊃ `Logistic` |
| swan/cwe79 | `SMO` | `RandomSubSpace` ⊃ `Logistic` |
| swan/cwe89 | `SMO` | `SimpleLogistic` |

The full descriptors, hyperparameters included, live in the `MODEL_TAG` case statement at the
top of each job script: `sscm.sh`, `dev-assist.sh`, `source.sh`, `sink.sh`, `sanitizer.sh`,
`cwe79.sh`, `cwe89.sh`.

### Stage 2: Repeated Cross-Validation

Stage 2 produced the RQ1 numbers and Table 3. Each experiment shuffles the training ARFF and runs 10-fold cross-validation for WEKA and MEKA. The
repetition is the seed list, so each model is evaluated on 100 different shuffles.

`experiments.cnf` defines the grid. Its `keyfields` are `seeds`, `dataset` and `classifier`, and
its `classifier` line names two descriptors, the `weave` model and the published one. That is 100
seeds by 2 models, so 200 experiments per case study and 1,400 in total.

The literal string `weave` is a placeholder in the `classifier` and `dataset` lines, substituted
at run time with the descriptor stage 1 selected and with `dataset.train`.
Results go to MySQL, one table per approach (`sscm`, `dev_assist`, `swan`), with the columns
`resultfields` lists. `analysis/inputs/model-selection-dump.sql` is a dump of those tables.

**To run it.** There is no stage-2-only flag. `-X` is explainer only, and without it the CLI runs
selection, then cross-validation, then the explainer in one job. Stage 2 therefore needs a MySQL
server matching `db.properties`, and it runs after a fresh selection whose result the pinned
`experiments.cnf` ignores.

```bash
mysql -e "CREATE DATABASE weave"     # db.properties expects localhost:3306, user root
java -jar cli/target/cli-1.0-jar-with-dependencies.jar \
  -t weka -c evaluation/ml4srm/sscm/sscm.properties
```

`synchronizeExperiments()` inserts the grid into the table on first run, then
`sequentiallyConductExperiments(-1)` executes the rows that have no result yet. An interrupted run
resumes instead of duplicating.

### Stage 3: Model Explanation

**Submit the SHAP job, once per model:**

```bash
sbatch --export=ALL,WEAVE_HOME=$WEAVE_HOME,MODEL_TAG=published evaluation/ml4srm/sscm.sh
sbatch --export=ALL,WEAVE_HOME=$WEAVE_HOME,MODEL_TAG=weave     evaluation/ml4srm/sscm.sh
```

`MODEL_TAG` picks both the classifier to explain and, through
`paths.output = .../explanations/${MODEL_TAG:-published}`, the directory the
results land in. It defaults to `published`.

Off the cluster, run it directly. The job scripts are thin wrappers around one invocation:

```bash
MODEL_TAG=weave java -XX:MaxRAMPercentage=75 \
  -jar target/cli-1.0-jar-with-dependencies.jar \
  -t weka -c evaluation/ml4srm/sscm/sscm.properties \
  -X -e "<the descriptor from sscm.sh>"
```

`-X` is explainer-only. It skips ML-Plan model selection and cross-validation, so no MySQL
server is needed. Dropping `-X` runs the full three-stage pipeline and *does* need the database
in `db.properties`. The committed values there are localhost placeholders.

After the run completes, the SLURM logs and explanations are exported.
Each `explanations/<tag>/` directory holds the `slurm-*.out` of the run that produced the
results beside it, which is the most recent successful run for that case study and model:

| Case study | `weave` | `published` |
|---|---|---|
| sscm | 34169430 | 34169429 |
| dev-assist | 33883587 | 33883588 |
| swan/source | 33855250 | 33855249 |
| swan/sink | 33855251 | 33855252 |
| swan/sanitizer | 33855254 | 33855253 |
| swan/cwe79 | 33855247 | 33855256 |
| swan/cwe89 | 33855246 | 33855245 |

`slurm-logs/` keeps the other 21 runs, the earlier successful runs that were
superseded plus the failures, mirroring the same
`<case-study>/<tag>/` layout.

## Reading the output

Both `explanations/published/` and `explanations/weave/` have the same shape. WEAVE
writes the two feature matrices at the top; the Python SHAP script writes
everything under `global/` and `local/`.

The five binary SWAN labels use the flat layout below. `sscm` (3 classes) and
`dev-assist` (5 labels) fan the same files out per class or per label. See
below.

```
explanations/<tag>/
├── global_features.csv          the training matrix, ARFF order
├── local_features.csv           the GXA matrix, ARFF order
├── slurm-<jobid>.out            the log of the run that produced this directory
├── global/
│   ├── global_beeswarm.pdf
│   ├── global_shap_values.csv       signed SHAP, one row per explained instance
│   └── global_shap_aggregated.csv   Feature, mean(|SHAP|)
└── local/
    ├── local_instance_<row>.pdf     waterfall plot, one per explained instance
    ├── local_shap_values.csv
    └── local_shap_aggregated.csv
```

For `dev-assist` (MEKA) everything is per label: `global_beeswarm_label_<l>.pdf`
plus a combined `global_beeswarm_all_labels.pdf`,
`global_shap_values_label_<l>.csv`, `global_shap_aggregated_all_labels.csv`, and
local plots split into `local/<l>/local_instance_<row>_label_<l>.pdf` alongside
`local/local_shap_aggregated_all_labels.csv`.

`sscm` has three class values, so the same files are written per class with a `_class_<k>`
suffix. The WEKA explainer covers every class when there are more than two:

```
explanations/<tag>/
├── global/
│   ├── class_index.csv                       class_index → class_name
│   ├── global_beeswarm_class_<k>.pdf
│   ├── global_beeswarm_all_classes.pdf       all classes pooled into one plot
│   ├── global_shap_values_class_<k>.csv
│   ├── global_shap_aggregated_class_<k>.csv
│   └── global_shap_aggregated_all_classes.csv
└── local/
    ├── <k>/local_instance_<row>_class_<k>.pdf
    ├── <k>/local_shap_values_class_<k>.csv
    ├── <k>/local_shap_aggregated_class_<k>.csv
    └── local_shap_aggregated_all_classes.csv
```

`class_index.csv` records which class each `<k>` is, so the output reads without going back to
the ARFF header. For `sscm` that is `0 = None`, `1 = Target`, `2 = Input`.

**Read it with `pd.read_csv(..., keep_default_na=False)`.** sscm's first class is literally
named `None`, which pandas otherwise parses as `NaN`.

A binary model still gets only the positive class, under the unsuffixed names above. That
loses nothing, since two probabilities sum to 1, so the negative class's SHAP values are the
negation of the positive's.

### Mapping a SHAP row back to a method

The `*_shap_values*.csv` files are the raw, signed SHAP matrices. The
`*_aggregated.csv` files are `mean(|SHAP|)` collapses of exactly this data and
cannot be un-collapsed. Their first four columns are the ones you need:

| Column | Meaning |
|---|---|
| `row` | position within the explained set; matches `local_instance_<row>.pdf` |
| `dataset_row` | row in `local_features.csv`, which is written in ARFF order and so is order-locked to the `*-gxa-methods` file |
| `base_value` | `E[f(x)]` over the background set |
| `prediction` | `base_value + Σ SHAP`, i.e. the model output being explained |

So `local_instance_7.pdf` → the `row = 7` line of `local_shap_values.csv` → its
`dataset_row` value *n* → line *n+1* of `<name>-gxa-methods`. The explained set
is a random subsample of the test set (seeded `default_rng(42)`), which is why
`row` and `dataset_row` differ.

Sampling is configured per case study and is identical across all seven:
500 background / 500 explained for global, 250 / 250 for local.

## Analysis

`analysis/` holds the code that turns the SHAP outputs above into the figures and tables the
paper reports.

```bash
analysis/reproduce.sh                                    # everything, about a minute
ML4SRM_FIGURES=/path/to/figures analysis/reproduce.sh    # collect output elsewhere
```

Dependencies are the same `requirements.txt` as the explainer, though the analysis needs only
`matplotlib` and `numpy`.

| Script | Produces |
|---|---|
| `make_shap_figs.py` | the RQ2 and RQ3 figures: `output/rq2/heat_*.pdf`, `slope_*.pdf` |
| `make_perf_tables.py` | the RQ1 per-class performance figures, `output/rq1/` |
| `cvlog.py` | not run directly; reads the cross-validation log for the other two. `python3 cvlog.py` prints what the dump contains |
| `make_tables.py` | the five tables the paper reports, as LaTeX, in `output/tables/` |

Each prints what it read and verified rather than failing silently, so read the output.
`make_shap_figs.py` also cross-checks its derived feature-group counts against the ARFF
headers, and reports a disagreement rather than carrying on.

### Data tables

`make_tables.py` writes one file per table. 

| File | Label in the paper | Built from |
|---|---|---|
| `table1_srm_sources.tex` | `table:srm-sources` | the SRM-Dataset catalog |
| `table2_feature_groups.tex` | `table:feature-groups` | the ARFF headers |
| `table3_selected_models.tex` | `table:selected-models` | `inputs/model-selection-dump.sql` |
| `table4_concentration.tex` | `table:concentration` | the `explanations/` directories |
| `table5_gxa_predictions.tex` | `table:gxa-predictions` | the `explanations/` directories and the GXA ARFFs |


**Table 1 needs the SRM-Dataset repository**, which owns the catalog. It defaults to a checkout
beside this one. Otherwise set `SRM_DATASET_HOME`. Without it, that table is skipped and the
other four are still written.

```bash
SRM_DATASET_HOME=/path/to/srm-dataset analysis/reproduce.sh
```


