# Improving Deep Learning for HAR with shallow LSTMs (ISWC 21' Note paper)

This is the official GitHub page of the note paper publication "Improving Deep Learning for HAR with shallow LSTMs" presented at the ISWC 21'.

## Repo Structure
- log_files: folder containing all log files of experiments mentioned in the paper
- job_scripts: contains files with terminal commands corresponding to the experiments mentioned in the paper
- data_processing: contains file for data processing (analysis, creation, preprocessing and sliding window approach)
- model: folder containing the DeepConvLSTM model, train and evaluation script
- main.py: main script which is to be run in order to commence experiments
- Results.xlsx: excel file containing all expererimental results presented in the paper

## Datasets

The datasets used during experiments can be either:
- Downloaded via this link
- Created using the dataset_creation.py file in the data_processing directory. See the Dataset Creation section for a how-to guide how to so.

The datasets need to be put in a seperate /data directory within the main directory of the repository in order for the main.py script to work.

### Dataset creation

Coming soon

## (Re-)running experiments

To run experiments one can either modify the main.py file (see hyperparameter settings in the beginning of the file or run the script via terminal and giving necessary hyperparameters via flags. See the main.py file for all possible hyperparameters which can be used. All terminal commands used during our experiments can be found in the corresponding job script file in the job_scripts folder. 

## Available log files of experiments

Note that within the log files accuracy, precision, recall and F1-score inbetween epochs were calculated using a 'weighted' and not 'macro' approach (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html for the difference between the two). Note that thus temporary results will be different if re-run, but final results will still be the same as reported in our publication.

## Results file

The results excel sheet (results.xlsx) contains all results mentioned within the paper as well as aggregated information about the standard deviation across runs, per-class results and standard deviation across subjects.

Tables and figures coming soon!

## Installing PyAV
- For Linux: 
```
git clone https://github.com/pscholl/PyAV
sudo apt-get install libx264-dev
sudo apt-get install libavformat-dev
sudo apt-get install libavdevice-dev
conda install cython
cd PyAV
./scripts/build-deps
make
  
