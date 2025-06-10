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

<h3> Prerequisites</h3>

<ul>
    <li> 
        Ensure python is installed in your machine:<br>
         <code>python -version</code> or <code>python3 -version</code> <br>
    </li>
</ul>

<h3> Creating and activating a virtual environment</h3>

Create a virtual environment to run the program. <br>
<code>python -m venv venv</code> or <code>python3 -m venv venv</code> <br>

To activate the virtual environment, use:<br>
Windows: <code> venv\Scripts\activate </code> <br>
Mac/Linux: <code> source venv/bin/activate </code>

<h3> Installing Dependencies</h3>
 Use pip to install the dependencies<br>
 <code> pip install -r requirements.txt </code>

<h3> Running the program</h3>
 Run the program using the python command after passing the train, test and output directories as arguments. <br>

```bash
python Dev-Assist.py --train <Path to the training file>\
--test <Path to the test ARFF file>\
--out <Output directory where logs and SHAP plots are saved> 
```

### Explainability for WEKA models ([weka](shap/weka))

### Setup

The global and local explainability of the WEKA based models used in the SWAN and SSCM approaches are evaluated. <br>

## Setup

Instructions to run the explainability experiments.

### Prerequisites

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

### Creating and activating a virtual environment

Create a virtual environment to run the program. <br>
<code>python -m venv venv</code> or <code>python3 -m venv venv</code> <br>

To activate the virtual environement, use:<br>
Windows: <code> venv\Scripts\activate </code> <br>
Mac/Linux: <code> source venv/bin/activate </code>

### Installing Dependencies
Use pip to install the dependencies<br>
<code> pip install -r requirements.txt </code>

### Running the program
Run the program using the python command after passing the train, test and output directories as arguments. <br>

```text
python sink.py --train <Path to the training file>\
--test <Path to the test ARFF file>\
--out <Output directory where logs and SHAP plots are saved> 
```