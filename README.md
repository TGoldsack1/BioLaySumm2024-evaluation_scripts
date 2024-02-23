# BioLaySumm Evaluation Scripts

This repository contains the scripts used for the evaluation of the BioLaySumm 2024 Shared Task.

The scripts are configured to run on the validation data provided to participants.
***
## Setup

Before running the scripts, you must download the AlignScore and LENS models. This can be done by running `get_models.sh` script:

```
bash ./get_models.sh
```


Additionally, you will need to install the other dependencies. The easiest way to do this is to use the provided `requirements.txt` file.

```
pip install -r requirements.txt
```


## Running Evaluation
Once setup is complete, you can run the evaluation script, `evaluate.py` on your predicted summaries.
It is advised to run the evaluation on a GPU due to the use of model-based metrics.
The script expects 2 positional arguments - the path to the directory containing the predicted summary text files (i.e., `elife.txt` and `plos.txt`) and the path to the directory containing provided validation `.jsonl` files: 

```
evaluate.py /path/to/predicted/summaries /path/to/validation/data
```


The script will output the evaluation results as `.txt` files within the current directory.

**Note**: Even when running the evaluation on a GPU, the runtime for the evaluation script will be relatively long due to: 1) the large number of documents in the validation set (PLOS in particular), and 2) the longer runtime of model-based metrics (SummaC in particular). To run a quicker proxy evaluation, we would recommend editing the `evaluate.py` file to reduce the evaluation samples used for PLOS/SummaC.
