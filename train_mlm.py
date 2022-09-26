import json
import argparse
import random
import copy

import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import os
from evaluate import f_beta_score
from trial_1_2 import perentry_prompting, PromptCreator, RELATIONS


def collect_mlm_data(model_name: str, top_k: int = 100, gpu_id: int = -1, suffix: str = '', use_softmax: bool = True,
                     prompts_per_rel: dict = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)


    prompt_creator = PromptCreator(prompt_fn=None, fixed_prompt_mode='trial')
    prompt_creator.prompt_templates = prompts_per_rel

    pipe = pipeline(
        task="fill-mask",
        model=model,
        tokenizer=tokenizer,
        top_k=top_k,
        device=gpu_id,
        use_softmax=use_softmax
    )

    key_priors = dict()

    existing_subjects = {'train': {rel: None for rel in RELATIONS},
                         'dev': {rel: None for rel in RELATIONS},
                         'test': {rel: None for rel in RELATIONS}
                         }

    # for subset in ['train', 'dev', 'test']:
    for subset in ['train', 'dev', 'test', 'silver_train100', 'silver_train200', 'silver_train500', 'silver_train1000',
                   'silver_train2000', 'silver_train']:
        print(f"Collecting data for {subset} set...")
        subj_already_exist_cnt = 0
        total_cnt = 0

        ofp = open(f'./data/{subset}_mlm{suffix}.jsonl', 'w', encoding='utf8')
        if subset == 'train':
            train2_ofp = open(f'./data/train2_mlm{suffix}.jsonl', 'w', encoding='utf8')
            dev2_ofp = open(f'./data/dev2_mlm{suffix}.jsonl', 'w', encoding='utf8')
        else:
            train2_ofp = None
            dev2_ofp = None

        with open(f'./data/{subset}.jsonl', 'r', encoding='utf8') as rfp:
            for lidx, line in enumerate(rfp):
                total_cnt += 1
                if lidx % 100 == 0:
                    print(f"lidx: {lidx}")
                entry = json.loads(line)
                out_lines = []

                if subset in ['train', 'dev', 'test']:
                    if existing_subjects[subset][entry['Relation']] is None:
                        existing_subjects[subset][entry['Relation']] = []
                    assert isinstance(entry['SubjectEntity'], str)
                    existing_subjects[subset][entry['Relation']].append(entry['SubjectEntity'])
                else:
                    subj_already_exist_flag = False
                    for ref_subset in existing_subjects:
                        assert isinstance(entry['SubjectEntity'], str)
                        if entry['SubjectEntity'] in existing_subjects[ref_subset][entry['Relation']]:
                            subj_already_exist_flag = True
                            break
                    if subj_already_exist_flag is True:
                        subj_already_exist_cnt += 1
                        continue

                _, final_prompts = perentry_prompting(entry, pipe, tokenizer.mask_token, key_priors,
                                                                   prompt_creator)

                curr_positives = entry['ObjectEntities'] if 'ObjectEntities' in entry else []

                for p, _ in final_prompts:
                    assert isinstance(p, str)
                    out_item = {'input_ids': p, 'posi_labels': curr_positives, 'Relation': entry['Relation']}
                                # 'negi_labels': curr_highlighted_negatives}
                    out_line = json.dumps(out_item, ensure_ascii=False)
                    out_lines.append(out_line)

                for ol in out_lines:
                    ofp.write(ol + '\n')

                if subset == 'train':
                    rho = random.random()
                    if rho < 0.8:
                        for ol in out_lines:
                            train2_ofp.write(ol + '\n')
                    else:
                        for ol in out_lines:
                            dev2_ofp.write(ol + '\n')
                else:
                    pass

        ofp.close()
        if subset == 'train':
            train2_ofp.close()
            dev2_ofp.close()

        print(f"subset {subset}; subj_already_exist_cnt: {subj_already_exist_cnt} / {total_cnt};")

        # print(f"{subset}: has {sst_posi_lbl_cnt} positive labels, {sst_negi_lbl_cnt} negative labels!")


class MLMDataset(Dataset):
    def __init__(self, in_fn, tokenizer: transformers.AutoTokenizer, extend_len: int, concentrated_rels: list = None):
        self.entries = []
        self.tokenizer = tokenizer
        mask_token = self.tokenizer.mask_token
        if isinstance(in_fn, list):
            fns = in_fn
        elif isinstance(in_fn, str):
            fns = [in_fn]
        else:
            raise AssertionError

        # From here on the input argument fn is overwritten!
        for curr_fn in fns:
            with open(curr_fn, 'r', encoding='utf8') as rfp:
                for line in rfp:
                    item = json.loads(line)
                    messy_temps = ['and northeastern',
                                   'and northern',
                                   'and parts of',
                                   'translation by Charles XIV',
                                   ]
                    for mt in messy_temps:  # manually disable the messy mined templates
                        if mt in item['input_ids']:
                            continue
                    if concentrated_rels is not None and item['Relation'] not in concentrated_rels:
                        continue
                    posi_labels_set = random.sample(item['posi_labels'], k=5) if len(item['posi_labels']) > 5 else item['posi_labels']
                    for posi_obj in posi_labels_set:

                        if len(posi_obj) == 0:
                            print(f"Posi-obj with zero surface forms!")
                            continue
                        posi_obj.sort(key=lambda x: len(x.split(' ')))
                        posi_obj = posi_obj[0]  # the shortest true object
                        input_words = item['input_ids'].replace(',', ' , ').replace('.', ' . ').replace(';', ' ; ').replace('!', ' ! ').replace('?', ' ? ').replace('  ', ' ').split(' ')
                        input_words = [x for x in input_words if len(x) > 0]
                        posi_obj_words = posi_obj.split()
                        mask_word_start = input_words.index(mask_token)
                        mask_word_end = mask_word_start + len(posi_obj_words)
                        instantiated = item['input_ids'].replace(mask_token, posi_obj)
                        mask_word_start = max(mask_word_start-extend_len, 0)
                        mask_word_end = min(mask_word_end+extend_len, len(instantiated))

                        instantiated = self.tokenizer(instantiated, add_special_tokens=True, padding=True)
                        instantiated_wordids = instantiated.word_ids()
                        instantiated = dict(instantiated)
                        for tidx in range(len(instantiated['input_ids'])):
                            if instantiated_wordids[tidx] is not None and mask_word_start <= instantiated_wordids[tidx] < mask_word_end:
                                masked_instantiated = copy.deepcopy(instantiated['input_ids'])
                                masked_instantiated[tidx] = self.tokenizer.mask_token_id
                                out_item = {}
                                out_item['labels'] = [x if i == tidx else -100 for i, x in enumerate(instantiated['input_ids'])]
                                out_item['input_ids'] = masked_instantiated
                                out_item['attention_mask'] = copy.deepcopy(instantiated['attention_mask'])
                                self.entries.append(out_item)
                        # out_item = copy.copy(item)
                        # out_item['curr_gold'] = posi_obj
                        # self.entries.append(out_item)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        # curr_entry = self.entries[idx]
        # tokens = self.tokenizer(self.entries[idx]['input_ids'], add_special_tokens=True, padding=True)
        # print(tokens.word_ids())
        # tokens = dict(tokens)
        # del tokens['token_type_ids']
        # label_tokid = self.tokenizer(self.entries[idx]['curr_gold'], add_special_tokens=False, padding=True, return_tensors='pt')['input_ids']  # shape: [1, 1]
        # if label_tokid.shape[1] > 1:
        #     for t in label_tokid[0]:
        #         print(self.tokenizer.decode(t))
        #     print(f"")
        # label_tokid = label_tokid[0][0].item()  # tensor(1)
        #
        # # print(f"torch version: {torch.__version__}")
        # labels = self.tokenizer(self.entries[idx]['input_ids'], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')['input_ids']
        # labels = torch.where(labels == self.tokenizer.mask_token_id, labels, -100)
        # labels = torch.where(labels == self.tokenizer.mask_token_id, label_tokid, labels)
        #
        # tokens['labels'] = labels
        # tokens['posi_labels'] = self.entries[idx]['posi_labels']
        # tokens['negi_labels'] = self.entries[idx]['negi_labels']

        return self.entries[idx]


def compute_metrics(predicts: transformers.EvalPrediction) -> dict:
    model_outs = predicts.predictions
    golds = predicts

    raise NotImplementedError


def train(model_name: str, train_fns: list, dev_fn: str, test_fn: str, concentrated_rels: list, ckpt_dir: str,
          logging_dir: str, lr: float, num_epochs: int, extend_len: int, no_cuda: bool):
    bert_config = transformers.AutoConfig.from_pretrained(model_name)
    bert_model = transformers.AutoModelForMaskedLM.from_pretrained(model_name, config=bert_config)
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    bert_collator = transformers.DataCollatorForTokenClassification(tokenizer=bert_tokenizer, padding=True, max_length=128)

    train_dataset = MLMDataset(train_fns, tokenizer=bert_tokenizer, extend_len=extend_len,
                               concentrated_rels=concentrated_rels)
    print(f"train dataset size: {len(train_dataset)}")
    # we extend the mask window only for training set, in evaluation, we only really care about the object tokens themselves;
    # In these cases, we set ``extend_len'' to 0
    dev_dataset = MLMDataset(dev_fn, tokenizer=bert_tokenizer, extend_len=0, concentrated_rels=concentrated_rels)
    print(f"dev dataset size: {len(dev_dataset)}")
    if test_fn is not None:
        test_dataset = MLMDataset(test_fn, tokenizer=bert_tokenizer, extend_len=0, concentrated_rels=concentrated_rels)
        print(f"test dataset size: {len(test_dataset)}")
    else:
        test_dataset = None
        print(f"test dataset not specified!")

    training_args = transformers.TrainingArguments(output_dir=ckpt_dir, overwrite_output_dir=False,
                                                   evaluation_strategy='epoch',
                                                   per_device_train_batch_size=32, per_device_eval_batch_size=64,
                                                   gradient_accumulation_steps=1, eval_accumulation_steps=8,
                                                   eval_delay=0, learning_rate=lr, weight_decay=0, num_train_epochs=num_epochs,
                                                   lr_scheduler_type='linear', warmup_ratio=0.1, log_level='debug',
                                                   logging_dir=logging_dir, logging_strategy='epoch',
                                                   save_strategy='epoch', save_total_limit=2, fp16=False,
                                                   dataloader_num_workers=0, auto_find_batch_size=False,
                                                   greater_is_better=False,
                                                   load_best_model_at_end=True, no_cuda=no_cuda)

    trainer = transformers.Trainer(model=bert_model, data_collator=bert_collator, train_dataset=train_dataset,
                                   args=training_args, eval_dataset=dev_dataset, tokenizer=bert_tokenizer, )
    # compute_metrics=compute_metrics)

    trainer.train()
    trainer.save_model(output_dir=os.path.join(ckpt_dir, 'best_ckpt'))
    dev_results = trainer.evaluate(dev_dataset)
    print(f"dev results: ")
    print(dev_results)
    if test_dataset is not None:
        test_results = trainer.evaluate(test_dataset)
        print(f"test results: ")
        print(test_results)

    return


def inference(inference_fn: str, ckpt_dir: str, mode: str):
    assert mode in ['eval']

    bert_config = transformers.AutoConfig.from_pretrained(ckpt_dir)
    bert_model = transformers.AutoModelForMaskedLM.from_pretrained(ckpt_dir, config=bert_config)
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt_dir)
    bert_collator = transformers.DataCollatorForTokenClassification(tokenizer=bert_tokenizer, padding=True,
                                                                    max_length=128)

    trainer = transformers.Trainer(model=bert_model, data_collator=bert_collator, tokenizer=bert_tokenizer, )

    if mode == 'eval':
        inference_dataset = MLMDataset(inference_fn, tokenizer=bert_tokenizer, extend_len=0)
        print(f"eval dataset size: {len(inference_dataset)}")
    else:
        raise AssertionError

    inference_results = trainer.predict(inference_dataset)
    print(inference_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='../lms/bert-large-cased')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--data_mode', type=str, default='development')
    parser.add_argument('--ckpt_dir', type=str, default='../lmkbc_checkpoints/mlm_checkpoints%s')
    parser.add_argument('--logging_dir', type=str, default='./mlm_tensorboard_logs%s')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--extend_len', type=int, default=2,
                        help='The number of tokens to extend in each direction beyond the masked object tokens, used to'
                             ' produce better representations of the masked object tokens for inference.')

    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--eval_subset', type=str, default='dev')
    parser.add_argument('--comment', type=str, default='',
                        help='comment to be appended to checkpoint names, if not set, will copy data_suffix to here.')

    # arguments below are for prompting, used in collecting training data
    # parser.add_argument('--thresholds_suffix', type=str, default='trial_1.2_blc_withsoftmax')
    # parser.add_argument('--hlt_ratio', type=float, default=0.5)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--collect_data_gpu_id', type=int, default=-1)
    parser.add_argument('--use_softmax', action='store_true')
    parser.add_argument('--prompt_style', type=str, default=None)
    # parser.add_argument('--use_sticky_thres', action='store_true')

    parser.add_argument('--data_suffix', type=str, default='')
    parser.add_argument('--thresholds_fn_feat', type=str, default='')
    parser.add_argument('--silver_data_size', type=str, default='')
    parser.add_argument('--concentrated_rels', type=str, nargs='*', default=None)

    args = parser.parse_args()

    args.comment = args.data_suffix + args.comment
    assert args.job_name in ['collect_data', 'train', 'eval', 'predict']

    if args.job_name == 'collect_data':
        # Build instantiated templates with the top-k templates which were empirically found to be best.
        thresholds_fn = f'./thresholds/thres_{args.thresholds_fn_feat}_F1.0.json'
        with open(thresholds_fn, 'r', encoding='utf8') as tfp:
            item = json.load(tfp)
            num_prompts = item['num_prompts']
            prompt_order = item['prompt_order']
            prompts_per_rel = {rel: [tuple(x[2]) for x in prompt_order[rel][:num_prompts[rel]]] for rel in RELATIONS}
        if args.data_suffix == '':
            args.data_suffix += '_' + args.thresholds_fn_feat
        else:
            args.data_suffix += args.thresholds_fn_feat
        collect_mlm_data(args.model_name, top_k=args.top_k, gpu_id=args.collect_data_gpu_id, suffix=args.data_suffix,
                         use_softmax=args.use_softmax, prompts_per_rel=prompts_per_rel)
    elif args.job_name == 'train':
        args.comment += f'_{args.data_mode}'
        if args.data_mode in ['silver_A', 'silver_B', 'silver_C']:
            assert args.silver_data_size is not None
            if args.silver_data_size == '':
                args.comment += '_sizeall'
            else:
                args.comment += f'_size{args.silver_data_size}'
            if args.concentrated_rels is not None:
                print(f"Concentrated rels: {args.concentrated_rels}")
                args.comment += f'_crels_' + '_'.join(args.concentrated_rels)
        args.ckpt_dir = args.ckpt_dir % args.comment
        args.logging_dir = args.logging_dir % args.comment

        if args.data_mode == 'development':
            train_fns = [os.path.join(args.data_dir, f'train2_mlm{args.data_suffix}.jsonl')]
            dev_fn = os.path.join(args.data_dir, f'dev2_mlm{args.data_suffix}.jsonl')
            test_fn = os.path.join(args.data_dir, f"dev_mlm{args.data_suffix}.jsonl")
        elif args.data_mode == 'submission':
            train_fns = [os.path.join(args.data_dir, f'train_mlm{args.data_suffix}.jsonl')]
            dev_fn = os.path.join(args.data_dir, f'dev_mlm{args.data_suffix}.jsonl')
            test_fn = os.path.join(args.data_dir, f"test_mlm{args.data_suffix}.jsonl")
        elif args.data_mode == 'silver_A':  # tests for using silver data alone
            train_fns = [os.path.join(args.data_dir, f'silver_train{args.silver_data_size}_mlm{args.data_suffix}.jsonl')]
            dev_fn = os.path.join(args.data_dir, f'train_mlm{args.data_suffix}.jsonl')
            test_fn = os.path.join(args.data_dir, f"dev_mlm{args.data_suffix}.jsonl")
        elif args.data_mode == 'silver_B':  # tests for using silver and real training data jointly, with dev set as test
            train_fns = [os.path.join(args.data_dir, f'silver_train{args.silver_data_size}_mlm{args.data_suffix}.jsonl'),
                         os.path.join(args.data_dir, f'train2_mlm{args.data_suffix}.jsonl')]
            dev_fn = os.path.join(args.data_dir, f'dev2_mlm{args.data_suffix}.jsonl')
            test_fn = os.path.join(args.data_dir, f"dev_mlm{args.data_suffix}.jsonl")
        elif args.data_mode == 'silver_C':  # using silver and real training data jointly, for submission
            train_fns = [os.path.join(args.data_dir, f'silver_train{args.silver_data_size}_mlm{args.data_suffix}.jsonl'),
                         os.path.join(args.data_dir, f'train_mlm{args.data_suffix}.jsonl')]
            dev_fn = os.path.join(args.data_dir, f'dev_mlm{args.data_suffix}.jsonl')
            test_fn = os.path.join(args.data_dir, f"test_mlm{args.data_suffix}.jsonl")
        else:
            raise AssertionError
        train(args.model_name, train_fns, dev_fn, test_fn, args.concentrated_rels, args.ckpt_dir, args.logging_dir,
              args.lr, args.num_epochs, args.extend_len, args.no_cuda)
    elif args.job_name in ['eval', 'predict']:
        inference_fn = os.path.join(args.data_dir, f'{args.eval_subset}_mlm.jsonl')
        inference(inference_fn, ckpt_dir=args.model_name, mode=args.job_name)
    else:
        raise AssertionError

    print(f"Finished.")
