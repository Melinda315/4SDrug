# 4SDrug
This repository contains a implementation of our "4SDrug: Symptom-based Set-to-set Small and Safe Drug Recommendation" accepted by KDD 2022.

## Environment Setup
1. Pytorch 1.2+
2. Python 3.6+
3. dill 0.3.4

## Guideline

### data

Please download MIMIC3 from the official website.

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

### Citation
If you find the code useful, please consider citing the following paper:
```
@inproceedings{tan20224SDrug,
  title={4SDrug: Symptom-based Set-to-set Small and Safe Drug Recommendation},
  author={Tan, Yanchao and Kong, Chengjun and Yu, Leisheng and Li, Pan and Chen, Chaochao and Zheng, Xiaolin and Hertzberg, Vicki S and Yang, Carl},
  booktitle={Proceedings of the 28th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2022}
}
```
