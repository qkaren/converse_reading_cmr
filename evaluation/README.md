# Evaluation

## Requirements
* Works fine for both Python 2.7 and 3.6
* Please **downloads** the following 3rd-party packages and save in a new folder `3rdparty`:
	* [**mteval-v14c.pl**](https://goo.gl/YUFajQ) to compute [NIST](http://www.mt-archive.info/HLT-2002-Doddington.pdf). You may need to install the following [perl](https://www.perl.org/get.html) modules (e.g. by `cpan install`): XML:Twig, Sort:Naturally and String:Util.
	* [**meteor-1.5**](http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz) to compute [METEOR](http://www.cs.cmu.edu/~alavie/METEOR/index.html). It requires [Java](https://www.java.com/en/download/help/download_options.xml).


## Create test data:

Please refer to the [data extraction page](https://github.com/qkaren/converse_reading_cmr/tree/master/data) to create the data. To create validation and test data, please run the following command:

```make -j4 valid test refs```

This will create the multi-reference file, along with followng four files:

* Validation data: ``valid.convos.txt`` and ``valid.facts.txt``
* Test data: ``test.convos.txt`` and ``test.facts.txt``

These files are in exactly the same format as ``train.convos.txt`` and ``train.facts.txt`` already explained [here](https://github.com/qkaren/converse_reading_cmr/tree/master/data). The only difference is that the ``response`` field of test.convos.txt has been replaced with the strings ``__UNDISCLOSED__``.

Notes: 
* The two validation files are optional and you can skip them if you want (e.g., no need to send us system outputs for them). We provide them so that you can run your own automatic evaluation (BLEU, etc.) by comparing the ``response`` field with your own system outputs. 
* Data creation should take about 1-4 days (depending on your internet connection, etc.). If you run into trouble creating the data, please contact us.

### Data statistics

Number of conversational responses: 
* Validation (valid.convos.txt): 4542 lines
* Test (test.convos.txt): 13440 lines

Due to the way the data is created by querying Common Crawl, there may be small differences between your version of the data and our own. To make pairwise comparisons between systems of each pair of participants, we will rely on the largest subset of the test set that is common to both participants.  **However, if your file test.convos.txt contains less than 13,000 lines, this might be an indication of a problem so please contact us immediately**.

## Prepare your system output for evaluation:

To create a system output for evaluation, keep the ``test.convos.txt`` and relace ``__UNDISCLOSED__`` with your own system output.

## Evaluation script:

**Note: The script (which is used in the paper) sub-samples a subset of the test data for evaluation.**

Steps:
1) Make sure you 'git pull' the latest changes, including changes in ../data.
2) cd to `../data` and type make. This will create the multi-reference file used by the metrics (`../data/test.refs`).
3) Install 3rd party software as instructed above (METEOR and mteval-v14c.pl).
5) Run the following command, where `[SUBMISSION]` is the submission file you want to evaluate: (same format as the one you submitted on Oct 8.)
```
python dstc.py -c [SUBMISSION] --refs ../data/test.refs
```

Important: the results printed by dstc.py might differ slightly from the official results, if part of your test set failed to download.
