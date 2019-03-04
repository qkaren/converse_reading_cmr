# Code for "Conversing by Reading: Contentful Neural Conversation with On-demand Machine Reading"

Confidential, please do not redistribute. The code will be released under an MIT License.

This package contains two independent codebases for (1) re-creating the dataset of our experiments and (2) the CMR model described in our submission. All this code will be made publicly and freely available with the final version of our paper, and will be hosted on github.

**Disclaimer: While we made significant efforts to document and test the code, this is a preliminary release and we will further improve it by the time of the final version.**

## Data

Since the dataset is extracted from Reddit and web crawls, we are not able to release the data directly, but we provide code to recreate the dataset exactly. The raw data is downloaded from sources that are static (i.e., Reddit [dump](http://files.pushshift.io/reddit/comments/) and [Common Crawl](http://commoncrawl.org/), so two independent runs of the data extraction script will produce exactly the same data, and this ensures repeatable experiments on the data we are contributing.

**How to run:** After moving into the `data`, data extraction consists of a single command (`make -j4`), but the [README](data/README.md) file gives details about software and packages to install and further information about the data. 

Notes:
* The full data extraction pipeline may take 1-5 days depending on the compute and internet speed.
* In some rare cases, data extraction output might slightly differ across runs (< 0.1%) due to 503 errors caused by Common Crawl. The final version will better handle these rare errors. 

## Model

TODO
