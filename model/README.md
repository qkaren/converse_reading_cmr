## Run the code

<<<<<<< HEAD
**1)Setup Environment**

Before training or processing data, run `sh launch_san_docker.sh` first to enter the docker.
(if you don't have docker on your machine, please run:
`sudo curl -sSL https://get.docker.com/ | sh`
)
=======
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

 If you don't have docker on your machine, please run:<br>
  `sudo curl -sSL https://get.docker.com/ | sh`

  If it is your first time to use docker, please refer the link for the usage:<br>
 `https://docs.docker.com/get-started/`
 
 Regrading PyTorch versions, please refer:<br>
  `https://pytorch.org/get-started/previous-versions/`
>>>>>>> a5891c3b6005cf0d4ee673aebd2953994b24e12d

**2)Process data (Skip this step if you jsut want to train on toy data)**

* Converting the raw text file to id json file for training, run `sh run_preprocessing.sh`

* The output_path is `/data/processed/full`. You can modify the output_path by changing `--data_dir`.



**3)To train the model**

* Run `run_san_data_weighted.sh` on toy data by fault.

* If you want to train on full data, please modify `--data_dir` from `/data/processed/toy` to `/data/processed/full` .