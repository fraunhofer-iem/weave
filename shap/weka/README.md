# Explainability - WEKA
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