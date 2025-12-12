# ML4SRM

<b>ML4SRM</b> uses the eXRM tool to evaluate and explore the explainability of three different ML based SRM classification approaches. <br>
The datasets used and results obtained from the experiment are available in this directory.

## Evaluation results [(evaluation)](evaluation/)

[evaluation](evaluation/) contains the datasets and results of previously evaluated models using WEAVE. These models are used to detect Security-Relevant Methods (SRMs). This contains the directory [ml4srm](evaluation/ml4srm/), which uses the WEAVE tool to evaluate and explore the explainability of three different ML based SRM classification approaches.
The datasets used and results obtained from the experiment are available in this directory.

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



