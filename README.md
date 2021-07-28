# Improving Deep Learning for HAR with shallow LSTMs (ISWC 21' Note paper)

This is the official GitHub page of the note paper publication "Improving Deep Learning for HAR with shallow LSTMs" presented at the ISWC 21'.

## Repo Structure
- log files: folder containing all log files of experiments mentioned in the paper
- job scripts: contains files with terminal commands corresponding to the experiments mentioned in the paper
- data preprocessing:
- model: folder containing the DeepConvLSTM model, train and evaluation script
- main.py: main script which is to be run in order to commence experiments
- Results.xlsx: excel file containing all expererimental results presented in the paper

## (Re-)running experiments

In order to rerun experiments, one needs to first 


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
  
