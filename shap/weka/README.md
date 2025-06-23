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
<code> pip install -r requirements.txt </code>

*You can also make use of our [script](create_env.sh) to create a virtual environment and install the dependencies.*
Move to the root directory and use the following commands.
```text
chmod +x create_env.sh
./create_env.sh
source venv/bin/activate
```

#### Configurations
The models, hyperparameters, train and test files for each approach/label is configured in the [config](shap/weka/config.json) file.<br>

#### Running the program
Run the program using the python command after passing the approach, label and output directory as arguments. <br>

```text
python weka_wrapper.py --approach <swan|sscm> [--label <label for the swan approach>] --out <Output directory where logs and SHAP plots are saved>
```

<code>--label</code> (Optional, for swan only) One of source, sink, sanitizer, CWE79, CWE89

<b>Example</b>
```text
python shap/weka/weka_wrapper.py --approach swan --label sink --out evaluation/ml4srm/swan
```