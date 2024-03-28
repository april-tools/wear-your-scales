# Automated mood disorder symptoms monitoring from multivariate time-series sensory data: Getting the full picture beyond a single number

This codebase was jointly developed by [Filippo Corponi](https://github.com/FilippoCMC) and [Bryan M. Li](https://github.com/bryanlimy). It is part of the paper "[Automated mood disorder symptoms monitoring from multivariate time-series sensory data: Getting the full picture beyond a single number](https://www.nature.com/articles/s41398-024-02876-1)", published in Nature Translational Psychiatry. If you find this code or any of the ideas in the paper useful, please consider starring this repository and citing:

```bibtex
@article{corponi2024automated,
  title={Automated mood disorder symptoms monitoring from multivariate time-series sensory data: Getting the full picture beyond a single number},
  author={Corponi, Filippo and Li, Bryan M and Anmella, Gerard and Mas, Ariadna and Pacchiarotti, Isabella and Valent{\'\i}, Marc and Grande, Iria and Benabarre, Antoni and Garriga, Marina and Vieta, Eduard and others},
  journal={Translational Psychiatry},
  volume={14},
  number={1},
  pages={161},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## Installation
- Create a new [conda](https://conda.io/en/latest/) environment with Python 3.10.
  ```bash
  conda create -n timebase python=3.10
  ```
- Activate `timebase` virtual environment
  ```bash
  conda activate timebase
  ```
- Install all dependencies and packages with `setup.sh` script, works on both Linus and macOS.
  ```bash
  sh setup.sh
  ```

## Data
- [data/README.md](data/README.md) details the structure of the dataset.
- Preprocess dataset. In the present work, time alignment was fixed to 0 `--time_alignment 0`, whereas window length was explored in powers of 2 (ranging from 2<sup>2</sup> to 2<sup>10</sup>). Below is an example command for with `--segment_length 64`. Please note that all the datasets corresponding to the different window lengths to be explored during tuning should be created in advance.
  ```bash
  python preprocess_ds.py --segment_length 64 --time_alignment 0 --output_dir data/preprocessed/ta0_sl64 --overwrite
  ```
  - please see `--help` for all available options.

## Mood Disorder Symptoms Inference

### 1) Deep Learning

Below is an example command to create a supervised neural network for the task at hand.

```bash
python train.py --batch_size 64 --epochs 200 --seed 1234 --min_epochs 50 --lr_patience 10 --dataset data/preprocessed/ta0_sl64 --emb_dim 128 --lr 0.001 --weight_decay 0.001 --verbose 1 --num_units 128 --dropout 0 --model bilstm --task_mode 0 --imb_mode 0 --emb_type 0 --output_dir runs/test --clear_output_dir --save_predictions --test_time
```
  - please see `--help` for all available options.

#### Hyperparameters tuning

Hyperparameters tuning relies on [Weight & Biases](https://docs.wandb.ai/). Once a sweep has been created on wandb website, tuning can be run as shown below, replacing `<sweep_id>` and `<group_name>` with the values created online on wandb.

```bash
python sweep.py --sweep_id <sweep_id> --wandb_group <group_name> --num_trials 100 --output_dir runs/test
```

Once tuning has been carried out and the set of hyperparameters associated with the heightest validation set performance has been identified, plug the appropriate hyperparameters in the command shown in [1) Deep Learning](#1-deep-learning). 

#### Post-hoc analyses

Assuming the best model output was saved to `runs/<best_model>`, in order to perform the post-hoc analyses on the best performing model, run:

```bash
python post_hoc_analysis.py --experiment_dir runs/<best_model>
```

To produce the graphical model on item residuals, once the above has run, run the following. This assumes [R programming language](https://www.r-project.org/about.html) has been install into the machine:

```bash
Rscript ResidualNetwork.R -f runs/<best_model>/residuals.csv
```

### 2) Classical Machine Learning

Below is an example command to tune and test random forest classifiers (one for each item in the Hamilton Depression Rating Scale and Young Mania Rating Scale).


```bash
python train_baseline.py --path2preprocessed data/preprocessed --model random_forest --output_dir runs/rf  --clear_output_dir
```

