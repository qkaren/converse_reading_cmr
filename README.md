We will clean up and document code in final version. 

## Run the code

**1)Process data**
* Run `sh run_preprocessing.sh`

**2) Start Docker**

Before training and testing, run `sh launch_san_docker.sh` first to enter the docker.
(if you don't have docker on your machine, please run:
`sudo curl -sSL https://get.docker.com/ | sh`
)

**3)Test the model**

* Download pre-trained model checkpoint from [here](https://drive.google.com/file/d/1Wm5VQriCaAF3l3C571y_XzhYp75BzZ9w/view?usp=sharing). Put the checkpoint under `./checkpoint`
* Run `sh run_san_data_weighted.sh` to test the model
* After running, you can find output files under './output' and './full_output' (in format for dstc evaluation)

**4)To train the model**

* Modify the script 'run_san_data_weighted.sh' by changing `--if_train 0` to `--if_train 1`
* Run `sh run.sh`

