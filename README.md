# eXRM
## Empirical Evaluation and Explainability Tool

<b>eXRM</b> is an empirical evaluation and explainability tool for evaluating and exploring the explainability of WEKA-based machine learning models. <br>
<b>eXRM</b> performs an empirical evaluation of ML models (typically selected via AutoML approaches) by performing repeated cross-validation of the model and averaging the metrics. <br>
The WEKA models are configured in python using python-weka-wrapper3 and use SHAP to explore the explainability of the models.

There are three main folders 
- [Evaluation (evaluation)](evaluation/)
- [Model evaluator (model-evaluator)](model-evaluator/)
- [Shap (shap)](shap/)

## [Evaluation (evaluation)](evaluation/)

Contains the datasets and results of previously evaluated models using eXRM. This contains the directory [ml4srm](evaluation/ml4srm/), which uses the eXRM tool to evaluate and explore the explainability of three different ML based SRM classification approaches.
The datasets used and results obtained from the experiment are available in this directory.

## [Model evaluator (model-evaluator)](model-evaluator/)

Explain

## [Shap (shap)](shap/)

Contains the scripts to run [meka](shap/meka) and [weka](shap/weka) models.

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

Instructions to run the explainability experiments.

<h2> Prerequisites</h2>

<ul>
    <li> 
        Ensure python is installed in your machine:<br>
        &emsp; <code>python -version</code> or <code>python3 -version</code> <br>
    </li>
    <li>
        Navigate to the required directory: <br>
        &emsp; E.g. <code>cd SWAN</code>
    </li>
</ul>

<h3> Creating and activating a virtual environment</h3>

&emsp; Create a virtual environment to run the program. <br>
&emsp; <code>python -m venv venv</code> or <code>python3 -m venv venv</code> <br>

&emsp; To activate the virtual environement, use:<br>
&emsp; Windows: <code> venv\Scripts\activate </code> <br>
&emsp; Mac/Linux: <code> source venv/bin/activate </code>

<h3> Installing Dependencies</h3>
&emsp; Use pip to install the dependencies<br>
&emsp; <code> pip install -r requirements.txt </code>

<h3> Running the program</h3>
&emsp; You can now run the program using the python command. <br>
&emsp; E.g.
&emsp; <code>python source.py</code> or <code> python3 source.py</code> <br>
&emsp; The output and log files will be saved in the <code>/out</code> directory.