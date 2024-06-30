# Experiments for paper "Zero Inflation as a Missing Data Problem: a Proxy-based Approach"

## Directory structure

The files included in this folder are
1. `sims.ipynb`
1. `sims.py`
1. `incompatibility_example.ipynb`
1. `ZI_clabsi-UAI_submission.ipynb`
1. `ZI_clabsi-corrected.ipynb`

## Requirements

### Python jobs

1. Using `conda`

    conda create -n zi_tests python=3.9 scipy numpy pandas
    conda activate zi_tests
    conda install -c conda-forge scikit-learn
    conda install -c ankurankan pgmpy

See https://pgmpy.org/started/install.html

### Requirements for numerical method

Use `autobounds` from Duarte et al. (2023) (https://doi.org/10.1080/01621459.2023.2216909) to calculate numerical bound.
It's best to use their docker image, see their paper's suppplement
for installation information.

If [docker](https://www.docker.com/get-started/) has already been installed,
then you only need to run this in `bash` environment:

    docker run -p 8888:8888 -it gjardimduarte/autolab:v4

## Exp 1: Bound validity

1. Set `compute_num_bound = True` in `sims.py`.
2. Set number of DGPs `N = 1000000`
2. From terminal, run

    python3 sims.py

## Exp 2: Comparison to numerical bound

1. Set `compute_num_bound = True` in `sims.py`.
2. Run

    python3 bound_code.py

to obtain `.pip` files and a `.csv` file.
The `.csv` file contains analytical bounds, while the `.pip` file contain optimization problems
to be solved with [`scip`](https://www.scipopt.org/doc/html/) to obtain numerical bounds.

2. Run docker image and record the CONTAINERID

    docker run -p 8888:8888 -t gjardimduarte/autolab:v4

3. Run scip in container until for 200s. This is because primal solution is often found after 2 mins

    docker exec -t CONTAINERID timeout 200 scip -f /path/to/pip/file/in/container/ > /path/to/output/file

4. Get result

    grep -E "^Primal Bound" /path/to/output/file

Note: Sometimes `scip` does not terminate correctly, so you will need to read the output files
for values of "Primal Bound" yourself.

## Exp 3: Incompatibility example

Explanation and instruction is in the notebook `incompatibility_example.ipynb`

## Exp 4: CLABSI experiment

### Code

1. In our UAI submission, we used `ZI_clabsi-corrected.ipynb` with the data pre-processing code `preprocessing_PatientData_UAI_submission.ipynb`. This has an error, see note `note-data-processing.md`.
2. The corrected code is in `ZI_clabsi-corrected.ipynb` and `preprocessing_PatientData-corrected.ipynb`. 

### Data
We cannot publish the dataset, as it would violate patient confidentiality,
and is thus not allowed by the IRB application approved by our institution.
