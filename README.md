# On Coverage of Test Suite Accuracy

In this project, we have provided an improvement to Test Suite Accuracy, a metric used to classify a pair of SQL queries as semantically equivalent or inequivalent. 
Specifically, we improve the process of neighbor query generation during inference to provide additional variations in neighbor SQL queries. This additional variation in the set of neighbor queries eliminates the false positive occurences in Test Suite Accuracy when dealing with numerical values. 

Download dataset from [Google Drive](https://drive.google.com/file/d/11RaGF2u1LtLWqirGTfltiSY_E1D4GHhR/view?usp=sharing)

To execute:

`python evaluation.py --gold sqlresults/dev_gold_experiment.txt --pred sqlresults/dev_preds_experiment.txt --db database --table tables.json --etype exec --plug_value`
