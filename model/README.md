We will clean up and document code in final version. 

## Run the code

**1)Process data**

Converting the raw text file to id json file for training:

* Run `sh run_preprocessing.sh`

The output file is `/data`. You can modify the output data_dir by changing `--data_dir`.

**2) Start Docker**

Before training and testing, run `sh launch_san_docker.sh` first to enter the docker.
(if you don't have docker on your machine, please run:
`sudo curl -sSL https://get.docker.com/ | sh`
)

**3)To train the model**

* Run 'run_san_data_weighted.sh'.

