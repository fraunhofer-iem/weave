## Empirical Evaluation and Explainability Tool---*eXRM*

<b>eXRM</b> is an empirical evaluation and explainability tool for evaluating and exploring the explainability of WEKA-based machine learning models. <br>
Emperical evaluation: <b>eXRM</b> performs an empirical evaluation of ML models (selected via AutoML approaches) by performing repeated cross-validation of the model and averaging the metrics. <br>
Explainability: The WEKA models are configured in python using python-weka-wrapper3 and use SHAP to explore the explainability of the models.

There are three main folders 
- Evaluation results [(evaluation)](evaluation/)
- Automated model selection using AutoML [(model-evaluator)](model-evaluator/)
- Explainability using SHAP [(shap)](shap/)

## Evaluation results [(evaluation)](evaluation/)

[evaluation](evaluation/) contains the datasets and results of previously evaluated models using eXRM. These models are used to detect Security-Relevant Methods (SRMs). This contains the directory [ml4srm](evaluation/ml4srm/), which uses the eXRM tool to evaluate and explore the explainability of three different ML based SRM classification approaches.
The datasets used and results obtained from the experiment are available in this directory.

## Automated model selection using AutoML [(model-evaluator)](model-evaluator/)

Program to run AutoML for the model selection and emperical evaluation of the selected model using the multiple cross-validation technique.

## Explainability using SHAP [(shap)](shap/)
Contains the scripts to run interpretability or explainability of the [meka](shap/meka) and [weka](shap/weka) models.

### Artifacts
To replicate the findings presented in our paper, please follow the instructions below.

Clone/download this repository.

Create a virtual environment to run the program.<br>

```bash
chmod +x create_env.sh
./create_env.sh
```

Run the script to reproduce the results for RQ2 and RQ3.
```bash
chmod +x RQ2_and_RQ3.sh
./RQ2_and_RQ3.sh.sh
```
#### Docker
You can also run our experiments on a docker container using the [Dockerfile](Dockerfile) provided. Mount a local directory along with the container to run the program and retrieve the results.

```bash
docker build -t weka-runner .
docker run --rm -v $(pwd):/app/evaluation/demo weka-runner
```



