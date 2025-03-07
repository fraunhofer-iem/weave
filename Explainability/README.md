<h1> Setup </h1>

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