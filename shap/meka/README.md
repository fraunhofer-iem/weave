# Explainability - Dev-Assist

The global and local explainability of the Dev-Assist approach is evaluated using the meka-converter approach. <br>

Multi-Label Approach: LabelPowerset <br>
Base Classifier: RandomForestClassifier <br>

## Setup

Instructions to run the experiments.

### Prerequisites

<ul>
    <li> 
        Ensure python is installed in your machine:<br>
         <code>python -version</code> or <code>python3 -version</code> <br>
    </li>
</ul>

### Creating and activating a virtual environment

 Create a virtual environment to run the program. <br>
 <code>python -m venv venv</code> or <code>python3 -m venv venv</code> <br>

 To activate the virtual environment, use:<br>
 Windows: <code> venv\Scripts\activate </code> <br>
 Mac/Linux: <code> source venv/bin/activate </code>

### Installing Dependencies
 Use pip to install the dependencies<br>
 <code> pip install -r requirements.txt </code>

### Running the program
 Run the program using the python command after passing the train, test and output directories as arguments. <br>

```text
python Dev-Assist.py --train <Path to the training file>\
--test <Path to the test ARFF file>\
--out <Output directory where logs and SHAP plots are saved> 
```