# MODI
Calculate the modelability index of a chemical dataset [1]

### Usage
There are 3 functions for calculating MODI, the core function `modi`, and two wrappers, `modi_from_sdf` and `modi_from_csv`

`modi` takes as and input a numpy array of descritpros and a numpy array of labels and can be easy added to any QSAR script

`modi_from_sdf` and `modi_from_csv` are wrappers that allow you to pass a file rather than a preloaded array of data. The functions will load the datafor you and calculate morgan fingerprint following some default setting that the user can change

This script will work for both multiclass and binary labeled data. Additionally, beside just getting overall MODI, you can also get the un-normalized MODI of each class, allowing you to get a rough idea of teh balacning of your dataset and its distribution

### Command line
This script can also be used as a stand alone command line call. using `modi.py -h` so see the list of parameter and options that can be used

## Reference
[1]: Alexander Golbraikh, Eugene Muratov, Denis Fourches, and Alexander Tropsha
Journal of Chemical Information and Modeling 2014 54 (1), 1-4
DOI: 10.1021/ci400572x
