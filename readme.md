# 4SDrug
This repository contains a implementation of our "Symptom-based Set-to-set Small and Safe Drug Recommendation".

## Environment Setup
1. Pytorch 1.2+
2. Python 3.6+
3. dill 0.3.4

## Guideline

### data

We provide a dataset MIMIC3 , which contains:

- Train set, validation set and test set and each contains patient records, including symptom sets, 
  diagnosis sets and drug sets(```data_train.pkl```, ```data_eval.pkl```, ```data_test.pkl```)
- Train symptom sets and drug sets derived from the train set, 
  where few sets are combined into batches and can satisfy our training requirements(```sym_train_50.pkl```, ```drug_train_50.pkl```)
- Voc file that used to calculate the number of unique symptoms and drugs, 
  and convert their indices into their codes(```voc_final.pkl```)
- DDI matrix(```ddi_A_final.pkl```)

### model

- The implementation of 4SDrug(```model.py```);
- The implementation of Attention Mechanism(```aggregation.py```)
- The implementation of RAdam Optimizer(```radm.py```)

### utils

Data input and preprocessing(```dataset.py```)

### eval

Indications of model performance(```metrics.py```)

## Example to run the codes
```
python main.py --dataset MIMIC3 --batch_size 50 --alpha 0.5 --beta 1.0
```
