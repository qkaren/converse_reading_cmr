## Run the code

**1)Setup Environment**

Before training or processing data, run `sh launch_san_docker.sh` first to enter the docker.
(if you don't have docker on your machine, please run:
`sudo curl -sSL https://get.docker.com/ | sh`
)

**2)Process data (Skip this step if you jsut want to train on toy data)**

* Converting the raw text file to id json file for training, run `sh run_preprocessing.sh`

* The output_path is `/data/processed/full`. You can modify the output_path by changing `--data_dir`.



**3)To train the model**

* Run `run_san_data_weighted.sh` on toy data by fault.

* If you want to train on full data, please modify `--data_dir` from `/data/processed/toy` to `/data/processed/full` .

