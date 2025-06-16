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

### Explainability for Dev-Assist ([meka](shap/meka))
The global and local explainability of the Dev-Assist approach is evaluated using the meka-converter approach. <br>

Multi-Label Approach: LabelPowerset <br>
Base Classifier: RandomForestClassifier <br>

### Setup
Instructions to run the experiments.

#### Prerequisites
<ul>
    <li> 
        Ensure python is installed in your machine:<br>
         <code>python -version</code> or <code>python3 -version</code> <br>
    </li>
</ul>

#### Creating and activating a virtual environment
Create a virtual environment to run the program. <br>
<code>python -m venv venv</code> or <code>python3 -m venv venv</code> <br>

To activate the virtual environment, use:<br>
Windows: <code> venv\Scripts\activate </code> <br>
Mac/Linux: <code> source venv/bin/activate </code>

#### Installing Dependencies
 Use pip to install the dependencies<br>
 <code> pip install -r shap/meka/requirements.txt </code>

#### Running the program
 Run the program using the python command after passing the train, test and output directories as arguments. <br>
```text
python shap/meka/Dev-Assist.py --train <train_file_dir> --test <test_file_dir> --out <output_dir>
```
<ul>
    <li>
        <code>--train</code> Path to the training ARFF file
    </li>
    <li>
        <code>--test</code> Path to the test ARFF file
    </li>
    <li>
         <code>--out</code> Directory to store logs and SHAP plots
    </li>
</ul>

<b>Example</b>
```text
python shap/weka/Dev-Assist.py --train evaluation/ml4srm/dev-assist/train/meka-code.arff\ 
--test evaluation/ml4srm/dev-assist/test/dev-assist-owasp-benchmark.arff\ 
--out evaluation/ml4srm/dev-assist
```

### Explainability for WEKA models ([weka](shap/weka))

#### Setup
The global and local explainability of the WEKA based models used in the SWAN and SSCM approaches are evaluated. <br>
Instructions to run the explainability experiments.

#### Prerequisites
<ol>
    <li> 
            Ensure python is installed:<br>
    <code>python -version</code> or <code>python3 -version</code> <br>
    </li>
    <li>
            Ensure JAVA_HOME is set. <br>
    <ul>
        <li>
                        To set JAVA_HOME:
            <code>export JAVA_HOME=$(/usr/libexec/java_home)</code> or go to <b>System Properties → Environment Variables</b>
        </li>
        <li>
                        To check: 
            <code> echo $JAVA_HOME</code> or <code>echo %JAVA_HOME%</code> <br>
        </li>
    </ul>
    </li>
</ol>

#### Creating and activating a virtual environment

Create a virtual environment to run the program. <br>
<code>python -m venv venv</code> or <code>python3 -m venv venv</code> <br>

To activate the virtual environement, use:<br>
Windows: <code> venv\Scripts\activate </code> <br>
Mac/Linux: <code> source venv/bin/activate </code>

#### Installing Dependencies
Use pip to install the dependencies<br>
<code> pip install -r shap/weka/requirements.txt </code>

#### Configurations
The models, hyperparameters, train and test files for each approach/label is configured in the [config](shap/weka/config.json) file.<br>

#### Running the program
Run the program using the python command after passing the approach, label and output directory as arguments. <br>

```text
python shap/weka/weka_wrapper.py --approach <swan|sscm> [--label <label>] --out <output_dir>
```
<ul>
    <li>
        <code>--approach</code> Classification approach (swan or sscm)
    </li>
    <li>
        <code>--label</code> (Optional, for swan only) One of source, sink, sanitizer, CWE79, CWE89
    </li>
    <li>
         <code>--out</code> Directory to store logs and SHAP plots
    </li>
</ul>

<b>Example</b>
```text
python shap/weka/weka_wrapper.py --approach swan --label sink --out evaluation/ml4srm/swan
```

