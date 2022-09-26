# Task-Specific Pre-training & Prompt Decomposition for KG Population with LMs

## Introduction

This is the official repository for the ISWC 2022 paper [Task-specific Pre-training and Prompt Decomposition for 
Knowledge Graph Population with Language Models](https://arxiv.org/abs/2208.12539), which is the winner of Track 1 in the 
[LM-KBC challenge at ISWC 2022](https://lm-kbc.github.io/).

This repository implements our experimented approaches for doing MLM training with augmented knowledge triples, doing prompt
decomposition for improved understanding of questions by BERT, prompt retrieval, and candidate selection. Repository structure
and scripts to use are described below.

## Repository Structure

- ***additional\_data/*** : contains the augmented knowledge triples retrieved from Wikidata;
- ***data/*** : contains the various data versions used in the paper;
- ***outputs/*** : contains the predictions of the models;
- ***prompts/*** : contains the retrieved prompts from Wikipedia 
(joined with prompts from [previous work](https://aclanthology.org/2020.tacl-1.28.pdf) and the manual prompts);
- ***thresholds/*** : each file in it contains thresholds per relation per prompt and optionally the sticky-ratios per prompt;
- ***baseline.py*** : challenge baseline;
- ***check_elements.py*** : sanity check, checks if the answers in ChemicalCompoundElements relation are indeed chemical elements;
- ***count_label_stats.py*** : stats checker, outputs the number of entries per relation in the dataset, and the number of answers for each question;
- ***count_mlm_lines_per_rel_silver.py*** : stats checker, outputs the number of subject-object pairs per relation in the augmented silver knowledge triples;
- ***evaluate.py*** : evaluation scripts for the challenge;
- ***file_io.py***: helper functions from the challenge;
- ***filter_mined_prompts.py***: filters the prompts mined from Wikipedia, joins them with prompts from 
[previous work](https://aclanthology.org/2020.tacl-1.28.pdf) and the manual prompts to produce the `_filtered` prompt-files;
- ***organize_added_data.py***: filters out silver entries whose subject is infelicitous and those all of whose objects are infelicitous;
organizes the remaining entries in single silver_train_XX.json files, where XX are the different sizes (sampled);
- ***playground.py***: playground for testing pytorch syntax;
- ***toy.py***: playground for understanding the behavior of LM on various toy examples;
- ***train_binary.py***: gathers the data and trains models for various binary classification tasks (including isalive / isindep(endent) classifiers and the object re-ranker);
- ***train_mlm.py***: gathers the data and trains models for MLM training;
- ***trial_1_2.py***: main script for the challenge, outputs the predictions from prompting various LMs (off-the-shelf vs. fine-tuned) with various prompts (manual / retrieved);

## Scripts to Use for MLM Training

### Challenge main functions
These are implemented in *trial_1_2.py*. There are four functionalities:
- ***search_thres***: searches for the best threshold for each relation and prompt;
- ***run_single***: runs on the found set of thresholds;
- ***find_weight***: finds the best weights for the set of retrieved prompts (based on already found sets of thresholds) (or the set of top-K-performing retrieved prompts);
- ***run_single_weight***: runs on the found set of weights;

There are the following relevant arguments worth mentioning:
- `--raw_model`: the default model to use when no other per-relation models are specified;
- `--version`: the version of current experiment;
- `--job_name`: one of the above four jobs to add;
- `--subset`: ?;
- `--comments`: ?;
- `--top_k`: the maximum number of prompt outputs to consider;
- `--threshold_fn`: path to the threshold file (see default);
- `--use_softmax`: whether to use softmax for the prompt outputs or remove it;
- `--exit_criterion`: the minimum scale of improvement by which we consider one method / hyper-parameter to be superior to another;
- `--relaxed_thres`: ?;
- `--beta`: the beta parameter for the F1 score;
- `--search_sticky`: whether to search for sticky-ratios or keep them 0;
- `--prompt_esb_mode`: ?;
- `--lw_XX`: these are hyper-parameters for prompt-weight learner;

For our main results we do not use the prompt-weight learner, so we only do `search_thres` and `run_single` jobs.
Example scripts are as follows:
- `python trial_1_2.py --version trial_1.2_blc --job_name search_thres --subset train --comments _withsoftmax_multilm --use_softmax 1 --gpu 0 \
  --prompt_esb_mode cmb`;
- `python trial_1_2.py --version trial_1.2_blc --job_name run_single --subset dev --comments _withsoftmax_multilm --use_softmax 1 --gpu 0`;

In `trial_1_2.py` line 1227-1245, we specify the models to use for each relation for our final results. 
For example, for the `ChemicalCompoundElement` relation, we use the MLM-trained model checkpoint trained with data from 
only this relation;

In `--version`, you can specify the prompts to use. There are the following options:
- None: use the prompts from our manually curated list (with decomposition in `StateSharesBorderState`);
- `baseline`: use the prompts from challenge baseline;
- `dpdmined`: use the prompts mined from dependency sub-trees in Wikipedia;
- `spanmined`: use the prompts mined from spans in Wikipedia;
- `cmbmined`: use the prompts mined from both sources.

In practice in our prompt retrieval experiments, we use `cmbmined`; in our final results we abandon prompt retrieval and 
use the manually curated prompts (namely no identifiers specified with this respect).

There are two other arguments: `--isalive_res_fn` and `--isindep_res_fn`. These are the paths to the results of the
isalive / isindep classifiers (see Section X below).

### For MLM Training

1. Gather knowledge triples for the respective relations from WikiData [here](https://query.wikidata.org/), and store them under 
*additional_data/*, with the following naming convention: *additional_data/RELATION.json* (our augmentations are readily available);
2. Run *organize_added_data.py* to filter out infelicitous entries and organize the remaining entries in *silver_train_XX.json* files, 
under *data/*;
3. (optional) Run *count_mlm_lines_per_rel_silver.py* to check the number of subject-object pairs per relation in the augmented silver knowledge triples;
4. (optional) If you want to use the prompts mined from Wikipedia, run *prompt-selection* first to find out the set of prompts to check (See section X below);
5. Run `collect_data` task in *train_mlm.py* to gather the data for MLM training; during this process, duplicate entries between
augmented entries and any subset of the original training data are removed, and the remaining entries are stored in 
files such as *data/dev2_mlm_trial_1.2_cmbmined_blc_joint2_withsoftmax_whatcmb.jsonl*; Example script: `python train_mlm.py --job_name collect_data --model_name ../lms/bert-large-cased --top_k 100 \
  --collect_data_gpu_id 0 --use_softmax --prompt_style trial --use_softmax \
  --thresholds_fn_feat trial_1.2_blc_withsoftmax`;
6. Run `train` task in *train_mlm.py* to train the MLM model; example script: 
   - Training with the challenge training set: `python train_mlm.py --job_name train --model_name ../lms/bert-large-cased --data_mode development \
  --lr 5e-6 --num_epochs 10 --extend_len 0 --comment _lr5e-6_10_0  (Current best setting: _lr5e-6_10_0, next best _lr5e-6_10_2;)`;
   - Training with the augmented training set: `python train_mlm.py --job_name train --model_name ../lms/bert-large-cased --data_mode silver_B \
  --lr 5e-6 --num_epochs 5 --extend_len 0 --data_suffix _trial_1.2_blc_withsoftmax \
  --comment _lr5e-6_10_0 --silver_data_size XXX (--concentrated_rels XXX)`;
   - The `--data_suffix` argument should be the same as the one used in the `collect_data` task;
   the `--concentrated_rels` argument is for training separated model checkpoints for different relations;
   - For training with augmented training set, there are three options for the `--data_mode` argument: `silver_A`, `silver_B`, and `silver_C`;
     - `silver_A` means using silver data alone, without any original training data;
     - `silver_B` means using silver and gold training data jointly, with dev2 set as dev for golds;
     - `silver_C` means using silver and gold training data jointly, with entire dev set as dev for golds;
   - The `--extend_len` option is for extending the sizes of MASKS in the MLM task, from the object tokens themselves to the tokens surrounding the object tokens;
7. The MLM-trained checkpoints are then used in the place of BERT-large-cased 

### For Prompt Retrieval / Selection

1. Retrieve prompts from Wikipedia following the instructions in [this previous work](https://aclanthology.org/2020.tacl-1.28.pdf);
2. Run `filter_mined_prompts` to filter out the infelicitous prompts;
3. Use these prompts in the `--version` argument in `trial_1_2.py` to run the prompt retrieval experiments.

### For IsAlive / IsIndependent Classifiers (Pre-condition classifiers)

Not shown to be effective, but we include the code for completeness. To be documented later.

### For Candidate Re-ranking

Not shown to be effective, but we include the code for completeness. To be documented later.

## Results

We present the final results (average Precision / Recall / F1) of our best model with ablations in the following table:

| Method                 | Precision | Recall | F1    |
|------------------------|-----------|--------|-------|
| Baseline               | 0.958     | 0.304  | 0.309 |
| + threshold tuning     | 0.488     | 0.455  | 0.447 |
| + prompt decomposition | 0.510     | 0.469  | 0.464 |
| + MLM Training         | 0.745     | 0.547  | 0.543 |
| + Sticky Ratios        | 0.767     | 0.551  | 0.546 |

For more results please refer to our paper.

## Citation

If you use this code, please cite our paper:

```
@misc{https://doi.org/10.48550/arxiv.2208.12539,
  doi = {10.48550/ARXIV.2208.12539},
  
  url = {https://arxiv.org/abs/2208.12539},
  
  author = {Li, Tianyi and Huang, Wenyu and Papasarantopoulos, Nikos and Vougiouklis, Pavlos and Pan, Jeff Z.},
  
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Task-specific Pre-training and Prompt Decomposition for Knowledge Graph Population with Language Models},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```
