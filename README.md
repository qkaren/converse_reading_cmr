# Code for "Conversing by Reading: Contentful Neural Conversation with On-demand Machine Reading"

ACL 2019 submission, anonymous authors.
Confidential, please do not redistribute.
The code will be released under an MIT License.

This package contains code for (1) re-creating the dataset of our experiments and (2) the CMR model described in our submission. All this code will be made publicly and freely available with the final version of our paper, and will be hosted on github.

**Disclaimer: While we made significant efforts to document and test the code, this is a preliminary release and we will further improve it by the time of the final version.**

The release contains two independent codebases and subfolders to handle data and model.

## Data

Since the dataset is extracted from Reddit and web crawls, we are not able to release the data directly, but we provide code to recreate the dataset exactly. The raw data is downloaded from sources that are static (i.e., Reddit [dump](http://files.pushshift.io/reddit/comments/) and [Common Crawl](http://commoncrawl.org/), so two independent runs of the data extraction script will produce exactly the same data, and this ensures repeatable experiments on the data we are contributing.

After moving into the `data`, data extraction consist of a single command (`make -j4`), but the [README](data/README.md) file gives details about software and packages to install and further information about the data. 

Notes:
* The full data extraction pipeline may take 1-5 days depending on the computer it is run on and the internet connection speed.
* In some rare cases, data extraction output might slightly differ across runs (< 0.1%) due to 403 errors returned by Common Crawl. We will identify these rare cases in the final version of the code. 

## Model

TODO
