## WEKA Automated eValuator & Explainer (WEAVE)

WEAVE is a tool for automated machine learning (AutoML), empirical evaluation, and eXplainability AI (XAI) for WEKA-based machine learning models. Given a dataset, WEAVE's pipeline performs: 

- _Model selection:_ uses the automated machine learning tool ML-Plan/ML2-Plan to search and select a machine learning model (pipeline and hyperparameters)
- _Empirical evaluation_: performs repeated cross-validation on the selected model using the jaicore-experimenter library
- _Explainability_: uses SHAP (SHapley Additive exPlanations) to explain the output of the model with global and local explanations

WEAVE supports three toolkits, WEKA (single-label), MEKA (multi-label) and
scikit-learn, that can be selected with the `-t` flag. The three stages are chained by a
single value: the WEKA-style command-line descriptor of the selected model
(e.g. `weka.classifiers.meta.Bagging -P 179 ... -W weka.classifiers.trees.RandomForest ...`),
which is what stage 1 produces and what stages 2 and 3 consume.

### How do I get started with WEAVE?

**Requirements:** Java 17+, Maven, and a Python 3 interpreter with the
dependencies in `evaluation/ml4srm/requirements.txt`. Stage 2 additionally needs a MySQL database.

**Build:**

```bash
mvn -q clean package     # → cli/target/cli-1.0-jar-with-dependencies.jar
```

**Run the full pipeline** (selection → evaluation → explanation):

```bash
java -jar cli/target/cli-1.0-jar-with-dependencies.jar -t weka -c path/to/config.properties
```

**Run the explainer only**, on a model you already have. `-X` skips selection
and cross-validation, so no database is needed:

```bash
java -jar cli/target/cli-1.0-jar-with-dependencies.jar -t meka -c path/to/config.properties \
  -X -e "meka.classifiers.multilabel.meta.EnsembleML -S 1 -I 10 -P 67 -W meka.classifiers.multilabel.PS -- ..."
```

| Flag | Meaning |
|---|---|
| `-t`, `--toolkit` | `weka`, `meka` or `scikit` (default `weka`) |
| `-c`, `--config` | path to the `.properties` pipeline configuration |
| `-X`, `--explainer-only` | skip model selection and evaluation |
| `-e`, `--explain` | the model descriptor to explain, required with `-X` |

### Configuration

The remaining pipeline configuration beyond those four flags comes from the `.properties` file. Every key
is required, no defaults are specified, and a missing key fails at startup. Path
values may use `${VAR}` and `${VAR:-default}` placeholders, expanded from the
environment, and relative paths are resolved against the config file's own
directory.

```properties
port                        # HTTP prediction server port; 0 lets the OS pick
dataset.train               # ARFF for training and global SHAP
dataset.test                # ARFF for local SHAP
mlplan.timeout              # total search budget, minutes
mlplan.timeout.node         # per-node budget, minutes
mlplan.timeout.candidate    # per-candidate budget, minutes
mlplan.seed                 # search seed
mlplan.cpu                  # cores available to the search
jaicore.databaseConfig      # db.properties (MySQL connection)
jaicore.experimentConfig    # experiments.cnf (keyfields, seeds, resultfields)
paths.explainer             # shap_explainer.py
paths.output                # directory for the CSVs and plots
paths.python                # Python interpreter
shap.global.bg.samples      # background set size, global
shap.global.exp.samples     # explained set size, global
shap.local.bg.samples       # background set size, local
shap.local.exp.samples      # explained set size, local
```

### How the explainability stage works

SHAP runs in Python but has to score the live JVM model, so WEAVE bridges the
two. `ModelExplainer` starts an embedded HTTP prediction server, exports the feature matrices
as `global_features.csv` and `local_features.csv`, then launches
`shap_explainer.py` as a subprocess. SHAP's `KernelExplainer` calls back into
the JVM through that endpoint for every perturbation. Output is a beeswarm plot per class or label for the global view, and one
waterfall plot per explained instance for the local view, each alongside the
signed SHAP matrix as CSV.

### Repository layout

```
cli/                     the Maven module; CliOptions → PipelineRunner is the whole control flow
  ...module/ModelSelector, ModelExperimenter, ModelExplainer
  ...module/explainer/   prediction server, per-toolkit prediction services, shap_explainer.py
evaluation/ml4srm/       replication package for the ML4SRM case studies (see its README for more info)
Dockerfile               containerised runtime
```


### Docker

The image mirrors the repository layout under `/app` and sets `WEAVE_HOME=/app`,
so the committed `.properties` files resolve unedited and the invocation inside
the container is the same as on the host. It copies the assembled jar rather
than building it, so run Maven first. The build context is the repository root:

```bash
mvn -q clean package
docker build -t weave .
docker run --rm -it -v "$PWD/results:/app/evaluation/ml4srm/sscm/explanations" weave

# then, inside the container:
MODEL_TAG=new java -jar /app/weave.jar -t weka \
  -c /app/evaluation/ml4srm/sscm/sscm.properties \
  -X -e "<model descriptor from evaluation/ml4srm/sscm.sh>"
```

The committed SHAP results and job logs are excluded from the image by
`.dockerignore`, since the container regenerates them into
`/app/evaluation/ml4srm/<case>/explanations/<MODEL_TAG>`.
