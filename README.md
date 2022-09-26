## Overview

Below are the scripts to train / run different modules of the system. The key component of task-specific pre-training goes 
under the section of MLM pre-training. Note that for information security reasons we are not yet able to upload the pre-trained
model checkpoints to network hard drives such as Dropbox at this stage; model checkpoints can be re-trained via the scripts provided below, please also check the arguments in the python files for more options, and the report for hyper-parameter settings; model checkpoints will be released shortly, subject to approval from internal information security checks. 

## Trial 1.2 

### Threshold Searching (thresholds and sticky_ratios)
- python trial_1_2.py --version trial_1.2_blc --job_name search_thres --subset train --comments _withsoftmax_multilm --use_softmax 1 --gpu 0 \
  --prompt_esb_mode cmb

### Single Run
- python trial_1_2.py --version trial_1.2_blc --job_name run_single --subset dev --comments _withsoftmax_multilm --use_softmax 1 --gpu 0

## MLM Training

### Data Collection

- python train_mlm.py --job_name collect_data --model_name ../lms/bert-large-cased --top_k 100 \
  --collect_data_gpu_id 0 --use_softmax --prompt_style trial --use_softmax \
  --thresholds_fn_feat trial_1.2_blc_withsoftmax

### Training
- python train_mlm.py --job_name train --model_name ../lms/bert-large-cased --data_mode development \
  --lr 5e-6 --num_epochs 10 --extend_len 0 --comment _lr5e-6_10_0  (Current best setting: _lr5e-6_10_0, next best _lr5e-6_10_2;)
- python train_mlm.py --job_name train --model_name ../lms/bert-large-cased --data_mode silver_B \
  --lr 5e-6 --num_epochs 5 --extend_len 0 --data_suffix _trial_1.2_blc_withsoftmax \
  --comment _lr5e-6_10_0 --silver_data_size XXX (--concentrated_rels XXX)

