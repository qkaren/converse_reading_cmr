# Code for "Conversing by Reading: Contentful Neural Conversation with On-demand Machine Reading"
<!--
Confidential, please do not redistribute. The code will be released under an MIT License. (What to do with this sentence?)
-->
This is the code for the following paper:

[Conversing by Reading: Contentful Neural Conversation with On-demand Machine Reading](https://www.aclweb.org/anthology/P19-1539)                         
*Lianhui Qin, Michel Galley, Chris Brockett, Xiaodong Liu, Xiang Gao, Bill Dolan, Yejin Choi, Jianfeng Gao; ACL 2019*

This package contains two independent codebases for (1) re-creating the dataset of our experiments and (2) the CMR model described in the paper. All this code is made publicly and freely available with the final version of our paper and will be hosted on github.

<!--
**Disclaimer:** *While we made significant efforts to document and test the code, this is a preliminary release and we will further improve it by the time of the final version.*
-->

## Data
You can **download** the raw data from [here](https://drive.google.com/file/d/1U9F2qf7mA39bPpqcnY0VarnYDQ1ygS6e/view?usp=sharing) directly. 

For the purpose of reproducbility, we also provide code to recreate our dataset (note that both running this code or downloading the above data will yield the same output). Since the raw sources to recreate the data are static (Reddit and Common Crawl dumps), this ensures the data output remains the same, making our experiments reproducible.  

**How to run:** After moving into the `data`, data extraction consists of a single command (`make -j4`), but the [README](data/README.md) file gives details about software and packages to install and further information about the data. 

Notes:
* The full data extraction pipeline may take 1-5 days, depending on compute power and internet speed;
* In some rare cases, data extraction output might have slight differences across runs (< 0.1% of the data) due to 503 errors returned by the Common Crawl server. The final version will better handle these rare cases.

## Model

We provide code to train and test with the CMR model, which is described in details in this [README](model/README.md).

## How do I cite Conversing by Reading?
```
@inproceedings{qin-etal-2019-conversing,
    title = "Conversing by Reading: Contentful Neural Conversation with On-demand Machine Reading",
    author = "Qin, Lianhui and Galley, Michel and Brockett, Chris and Liu, Xiaodong 
              and Gao, Xiang and Dolan, Bill and Choi, Yejin and Gao, Jianfeng",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = "jul",
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1539",
    pages = "5427--5436",
}
```


