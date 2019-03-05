## Run the code

This directory contains code to train and test the CMR model. We provide two experimental conditions:
* **Toy data**: This one does NOT require generating the data using scripts `../data`, as a toy dataset is already provided. This only requires executing the steps (1) and (3) below. This lets the user quickly run the entire pipeline end-to-end to more easily understand how it works. This is of course only useful for understanding purposes, *as the small size of the training data will of course cause the generated outputs to be of very poor quality.*
* **Full data**: This runs on the full data, which requires first running scripts `../data` according to the instructions in that folder. This creates files in `data/raw/full`. After that raw data has been created, please run all the steps listed below.

**1) Setup Environment**
1. python3.6
2. install requirements:
   > pip install -r requirements.txt
3. You might need to download the en module for spacy
   > python -m spacy download en              # default English model (~50MB) <br>
   > python -m spacy download en_core_web_md  # larger English model (~1GB)
   
  Or pull our published docker: allenlao/pytorch-allennlp-rt
 <br>
 **Hints:**<br>

  If it is your first time to use docker, please refer the link for the usage:<br>
 `https://docs.docker.com/get-started/`
 
 Regrading PyTorch versions, please refer:<br>
  `https://pytorch.org/get-started/previous-versions/`

**2) Process data (Skip this step if you just want to train on toy data)**

* Converting the raw text file to id json file for training, run `sh run_preprocessing.sh`

* The output_path is `/data/processed/full`. You can modify the output_path by changing `--data_dir`.

**3) To train the model**

* Run `run_san_data_weighted.sh` on toy data by fault.

* If you want to train on full data, please modify `--data_dir` from `/data/processed/toy` to `/data/processed/full` .
