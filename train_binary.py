import json
import argparse
import random

import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from trial_1_2 import perentry_prompting, gather_key_priors, filter_objs, MODELIDS_REL_MAPPING, read_lm_kbc_jsonl, \
    PromptCreator
import os
from evaluate import f_beta_score, evaluate_per_sr_pair, combine_scores_per_relation, clean_object
import pandas as pd


def collect_data_isalive():
    for subset in ['train', 'dev', 'test']:

        ofp = open(f'./data/{subset}_isalive.jsonl', 'w', encoding='utf8')
        if subset == 'train':
            train2_ofp = open(f'./data/train2_isalive.jsonl', 'w', encoding='utf8')
            dev2_ofp = open(f'./data/dev2_isalive.jsonl', 'w', encoding='utf8')
        else:
            train2_ofp = None
            dev2_ofp = None

        with open(f'./data/{subset}.jsonl', 'r', encoding='utf8') as rfp:
            for line in rfp:
                item = json.loads(line)
                if item['Relation'] in ['PersonPlaceOfDeath', 'PersonCauseOfDeath']:
                    if 'ObjectEntities' not in item:  # as is the case for test set
                        lbl = None
                    elif len(item['ObjectEntities']) == 0:
                        lbl = True
                    else:
                        lbl = False
                    prompt = f"{item['SubjectEntity']} is still alive."
                    out_item = {'sent': prompt, 'lbl': lbl}
                    out_line = json.dumps(out_item, ensure_ascii=False)
                    ofp.write(out_line+'\n')
                    if subset == 'train':
                        rho = random.random()
                        if rho < 0.8:
                            train2_ofp.write(out_line+'\n')
                        else:
                            dev2_ofp.write(out_line+'\n')
                    else:
                        pass

        ofp.close()
        if subset == 'train':
            train2_ofp.close()
            dev2_ofp.close()


def collect_data_isindep():
    for subset in ['train', 'dev', 'test']:

        ofp = open(f'./data/{subset}_isindep.jsonl', 'w', encoding='utf8')
        if subset == 'train':
            train2_ofp = open(f'./data/train2_isindep.jsonl', 'w', encoding='utf8')
            dev2_ofp = open(f'./data/dev2_isindep.jsonl', 'w', encoding='utf8')
        else:
            train2_ofp = None
            dev2_ofp = None

        with open(f'./data/{subset}.jsonl', 'r', encoding='utf8') as rfp:
            for line in rfp:
                item = json.loads(line)
                if item['Relation'] in ['CompanyParentOrganization']:
                    if 'ObjectEntities' not in item:
                        lbl = None
                    elif len(item['ObjectEntities']) == 0:
                        lbl = True
                    else:
                        lbl = False
                    prompt = f"{item['SubjectEntity']} is not owned by another company."
                    out_item = {'sent': prompt, 'lbl': lbl}
                    out_line = json.dumps(out_item, ensure_ascii=False)
                    ofp.write(out_line+'\n')
                    if subset == 'train':
                        rho = random.random()
                        if rho < 0.8:
                            train2_ofp.write(out_line+'\n')
                        else:
                            dev2_ofp.write(out_line+'\n')
                    else:
                        pass

        ofp.close()
        if subset == 'train':
            train2_ofp.close()
            dev2_ofp.close()


# Reranking data is collected based on the mlm data: most confidently
def collect_data_rerank(model_names: list, model_rel_mapping: dict, top_k: int = 100, curr_thresholds: dict = None,
                        curr_sticky_ratios: dict = None, hlt_ratio: float = 0.5, suffix: str = '', use_softmax: bool = True,
                        prompt_style: str = None, tagger_isalive_fn: str = None, tagger_isindep_fn: str = None,
                        mask_token: str = '[MASK]'):
    if prompt_style == 'baseline':
        prompt_creator = PromptCreator(prompt_fn=None, fixed_prompt_mode='baseline')
    elif prompt_style == 'dpdmined':
        prompt_creator = PromptCreator(prompt_fn='./prompts/dependency_based_prompts_filtered.jsonl', fixed_prompt_mode=None)
    elif prompt_style == 'spanmined':
        prompt_creator = PromptCreator(prompt_fn='./prompts/middle_word_prompts_filtered.jsonl', fixed_prompt_mode=None)
    elif prompt_style == 'cmbmined':
        prompt_creator = PromptCreator(prompt_fn='./prompts/combined_mined_prompts_filtered.jsonl', fixed_prompt_mode=None)
    elif prompt_style == 'trial':
        prompt_creator = PromptCreator(prompt_fn=None, fixed_prompt_mode='trial')
    else:
        raise AssertionError

    pipes = []
    for midx, model_name in enumerate(model_names):
        if model_name is None:
            for rel in model_rel_mapping:
                model_rel_mapping[rel] = 1 - midx
            pipes.append(None)
            continue
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        assert mask_token == tokenizer.mask_token

        pipe = pipeline(
            task="fill-mask",
            model=model,
            tokenizer=tokenizer,
            top_k=top_k,
            device=midx if torch.cuda.is_available() else -1,
            use_softmax=use_softmax
        )
        pipes.append(pipe)

    key_priors = gather_key_priors(pipes, model_rel_mapping, prompt_creator=prompt_creator, mask_token=mask_token)

    for subset in ['train', 'dev', 'test']:
        print(f"Collecting data for {subset} set...")

        sst_posi_lbl_cnt = 0
        sst_negi_lbl_cnt = 0

        if tagger_isalive_fn is not None:
            tagger_isalive_fn = tagger_isalive_fn % subset
            with open(tagger_isalive_fn, 'r', encoding='utf8') as fp:
                tagger_isalive_results = json.load(fp)
        else:
            tagger_isalive_results = None
        isalive_cnt = 0

        if tagger_isindep_fn is not None:
            tagger_isindep_fn = tagger_isindep_fn % subset
            with open(tagger_isindep_fn, 'r', encoding='utf8') as fp:
                tagger_isindep_results = json.load(fp)
        else:
            tagger_isindep_results = None
        isindep_cnt = 0

        if subset == 'train':
            train2_ofp = open(f"./data/train2_rerank{suffix}.jsonl", 'w', encoding='utf8')
            dev2_ofp = open(f"./data/dev2_rerank{suffix}.jsonl", 'w', encoding='utf8')

        else:
            train2_ofp = None
            dev2_ofp = None

        ofp = open(f"./data/{subset}_rerank{suffix}.jsonl", 'w', encoding='utf8')

        with open(f'./data/{subset}.jsonl', 'r', encoding='utf8') as rfp:
            for lidx, line in enumerate(rfp):
                if lidx % 100 == 0:
                    print(f"lidx: {lidx}")

                entry = json.loads(line)
                out_pairs = []

                if subset in ['train', 'dev']:
                    posi_sf_set = set()
                    for posi_synset in entry['ObjectEntities']:
                        posi_sf_set.update(posi_synset)
                    if subset == 'train':
                        for sf in posi_sf_set:
                            out_pairs.append((sf, True))
                        sst_posi_lbl_cnt += len(posi_sf_set)
                    else:
                        pass
                elif subset in ['test']:
                    posi_sf_set = None
                else:
                    raise AssertionError

                if tagger_isalive_results is not None and entry['Relation'] in ['PersonPlaceOfDeath', 'PersonCauseOfDeath']:
                    print(
                        f"entry subject entity: {entry['SubjectEntity']}; isalive tagger results prompt: {tagger_isalive_results[isalive_cnt]}; should be aligned!")
                    tagger_isalive_val = tagger_isalive_results[isalive_cnt][1]
                    isalive_cnt += 1
                else:
                    tagger_isalive_val = None
                if tagger_isindep_results is not None and entry['Relation'] in ['CompanyParentOrganization']:
                    print(
                        f"entry subject entity: {entry['SubjectEntity']}; isindep tagger results prompt: {tagger_isindep_results[isindep_cnt]}; should be aligned!")
                    tagger_isindep_val = tagger_isindep_results[isindep_cnt][1]
                    isindep_cnt += 1
                else:
                    tagger_isindep_val = None

                joined_outputs, final_prompts = perentry_prompting(entry, pipes[model_rel_mapping[entry['Relation']]], mask_token, key_priors,
                                                                   prompt_creator=prompt_creator, tagger_isalive_val=tagger_isalive_val,
                                                                   tagger_isindep_val=tagger_isindep_val)
                curr_outputs, _ = filter_objs(joined_outputs, curr_thresholds[entry['Relation']] * hlt_ratio,
                                              entry['Relation'], sticky_ratio=curr_sticky_ratios[entry['Relation']])
                curr_outputs = [clean_object(x) for x in curr_outputs]

                if subset in ['test']:
                    assert posi_sf_set is None
                    for x in curr_outputs:
                        out_pairs.append((x, None))
                elif subset in ['train']:
                    curr_highlighted_negatives = [x for x in curr_outputs if x not in posi_sf_set]
                    for x in curr_highlighted_negatives:
                        out_pairs.append((x, False))
                    sst_negi_lbl_cnt += len(curr_highlighted_negatives)
                elif subset in ['dev']:
                    for obj in curr_outputs:
                        if obj in posi_sf_set:
                            out_pairs.append((obj, True))
                            sst_posi_lbl_cnt += 1
                        else:
                            out_pairs.append((obj, False))
                            sst_negi_lbl_cnt += 1
                else:
                    raise AssertionError

                rho = random.random()

                for obj, lbl in out_pairs:
                    for p in final_prompts:
                        osent = p.replace(mask_token, obj)
                        out_item = {'SubjectEntity': entry['SubjectEntity'], 'Relation': entry['Relation'],
                                    'ObjectEntity': obj, 'sent': osent, 'lbl': lbl}
                        out_line = json.dumps(out_item, ensure_ascii=False)
                        ofp.write(out_line+'\n')
                        if subset == 'train':
                            if rho < 0.8:
                                train2_ofp.write(out_line+'\n')
                            else:
                                dev2_ofp.write(out_line+'\n')
                        else:
                            pass
        ofp.close()
        if subset == 'train':
            train2_ofp.close()
            dev2_ofp.close()
        print(f"Subset {subset}: posi labels: {sst_posi_lbl_cnt}; negi labels: {sst_negi_lbl_cnt};")

def collect_data_from_trial_output(output_prefix: str):
    assert 'relaxed' in output_prefix
    for subset in ['train', 'dev', 'test']:

        sst_posi_lbl_cnt = 0
        sst_negi_lbl_cnt = 0

        ofp = open(f'./data/{subset}_rerank_{output_prefix}.jsonl', 'w', encoding='utf8')
        if subset == 'train':
            train2_ofp = open(f'./data/train2_rerank_{output_prefix}.jsonl', 'w', encoding='utf8')
            dev2_ofp = open(f'./data/dev2_rerank_{output_prefix}.jsonl', 'w', encoding='utf8')
        else:
            train2_ofp = None
            dev2_ofp = None

        with open(f'./data/{subset}.jsonl') as gold_fp, \
                open(f'./outputs/{output_prefix}_{subset}.jsonl', 'r', encoding='utf8') as predict_fp:
            for gold_line, predict_line in zip(gold_fp, predict_fp):

                gold_item = json.loads(gold_line)
                predict_item = json.loads(predict_line)
                assert gold_item['SubjectEntity'] == predict_item['SubjectEntity']
                assert gold_item['Relation'] == predict_item['Relation']
                curr_prompts = predict_item['Prompt']

                curr_pairs = []

                if subset in ['test']:
                    for predict_obj in predict_item['ObjectEntities']:
                        curr_pairs.append((predict_obj, False))
                else:
                    posi_sf_set = set()
                    for posi_synset in gold_item['ObjectEntities']:
                        for surface_form in posi_synset:
                            posi_sf_set.add(surface_form)
                    if subset in ['train']:
                        for surface_form in posi_sf_set:
                            curr_pairs.append((surface_form, True))
                        sst_posi_lbl_cnt += len(posi_sf_set)
                        for predict_obj in predict_item['ObjectEntities']:
                            if predict_obj in posi_sf_set:
                                # This means it is a true positive, they have already been covered above, so we continue
                                continue
                            curr_pairs.append((predict_obj, False))
                            sst_negi_lbl_cnt += 1
                    elif subset in ['dev']:
                        for predict_obj in predict_item['ObjectEntities']:
                            predict_obj = clean_object(predict_obj)
                            if predict_obj in posi_sf_set:
                                curr_pairs.append((predict_obj, True))
                                sst_posi_lbl_cnt += 1
                            else:
                                curr_pairs.append((predict_obj, False))
                                sst_negi_lbl_cnt += 1

                rho = random.random()
                for sf, lbl in curr_pairs:
                    for p in curr_prompts:
                        osent = p.replace(sf)
                        curr_outitem = {'SubjectEntity': gold_item['SubjectEntity'], 'Relation': gold_item['Relation'],
                                              'ObjectEntity': sf, 'sent': osent, 'lbl': lbl}
                        ol = json.dumps(curr_outitem, ensure_ascii=False)
                        ofp.write(ol + '\n')
                        if subset == 'train':
                            if rho < 0.8:
                                train2_ofp.write(ol + '\n')
                            else:
                                dev2_ofp.write(ol + '\n')
                        else:
                            pass

        ofp.close()
        if subset == 'train':
            train2_ofp.close()
            dev2_ofp.close()
        print(f"Subset {subset}: posi labels: {sst_posi_lbl_cnt}; negi labels: {sst_negi_lbl_cnt};")


class BinaryDataset(Dataset):
    def __init__(self, fn: str, use_label: bool, tokenizer: transformers.AutoTokenizer):
        self.entries = []
        self.use_label = use_label
        self.tokenizer = tokenizer
        with open(fn, 'r', encoding='utf8') as rfp:
            for line in rfp:
                item = json.loads(line)
                self.entries.append(item)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.entries[idx]['sent'], add_special_tokens=True, padding=True, truncation=True,
                                max_length=128)
        tokens = dict(tokens)
        del tokens['token_type_ids']
        tokens['sent'] = self.entries[idx]['sent']
        if self.use_label is True:
            assert isinstance(self.entries[idx]['lbl'], bool)
            assert self.entries[idx]['lbl'] is not None
            tokens['labels'] = 1 if self.entries[idx]['lbl'] is True else 0
        else:
            pass
        if 'SubjectEntity' in self.entries[idx]:
            tokens['SubjectEntity'] = self.entries[idx]['SubjectEntity']
            tokens['Relation'] = self.entries[idx]['Relation']
            tokens['ObjectEntity'] = self.entries[idx]['ObjectEntity']
        else:
            pass
        return tokens


def compute_metrics(predicts: transformers.EvalPrediction) -> dict:
    model_outs = predicts.predictions
    golds = predicts.label_ids
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for o, g in zip(model_outs, golds):
        o = torch.softmax(torch.tensor(o), dim=-1)
        o = 0 if o[0] > o[1] else 1
        if o == 1 and g == 1:
            tp +=1
        elif o == 1 and g == 0:
            fp += 1
        elif o == 0 and g == 0:
            tn += 1
        elif o == 0 and g == 1:
            fn += 1
        else:
            raise AssertionError

    prec = tp / (tp + fp) if (tp+fp) > 0 else 0
    rec = tp / (tp + fn)
    f1 = f_beta_score(prec, rec, beta=args.beta)
    acc = (tp + tn) / (tp + fp + tn + fn)
    print(f"tp: {tp}; fp: {fp}; tn: {tn}; fn: {fn};")
    return {'prec': prec, 'rec': rec, 'f1': f1, 'acc': acc}


def train(model_name: str, train_fn: str, dev_fn: str, test_fn: str, ckpt_dir: str, logging_dir: str, lr: float,
          num_epochs: int, no_cuda: bool, ):

    bert_config = transformers.AutoConfig.from_pretrained(model_name, num_labels=2)
    bert_model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, config=bert_config)
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    bert_collator = transformers.DataCollatorWithPadding(tokenizer=bert_tokenizer, padding=True, max_length=128)

    train_dataset = BinaryDataset(train_fn, use_label=True, tokenizer=bert_tokenizer)
    dev_dataset = BinaryDataset(dev_fn, use_label=True, tokenizer=bert_tokenizer)
    if test_fn is not None:
        test_dataset = BinaryDataset(test_fn, use_label=True, tokenizer=bert_tokenizer)
    else:
        test_dataset = None

    training_args = transformers.TrainingArguments(output_dir=ckpt_dir, overwrite_output_dir=False,
                                                   evaluation_strategy='epoch',  # eval_steps=20,
                                                   per_device_train_batch_size=16, per_device_eval_batch_size=32,
                                                   gradient_accumulation_steps=1, eval_accumulation_steps=8,
                                                   eval_delay=0, learning_rate=lr, weight_decay=0, num_train_epochs=num_epochs,
                                                   lr_scheduler_type='linear', warmup_ratio=0.1, log_level='debug',
                                                   logging_dir=logging_dir, logging_strategy='epoch',  # logging_steps=20,
                                                   save_strategy='epoch', save_total_limit=2, fp16=False,
                                                   dataloader_num_workers=0, metric_for_best_model='acc',
                                                   auto_find_batch_size=False, load_best_model_at_end=True,
                                                   no_cuda=no_cuda)

    trainer = transformers.Trainer(model=bert_model, data_collator=bert_collator, train_dataset=train_dataset,
                                   args=training_args,
                                   eval_dataset=dev_dataset, tokenizer=bert_tokenizer, compute_metrics=compute_metrics)

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


def inference(inference_fn: str, ckpt_dir: str, mode: str, result_fn: str, subm_ofn: str, orig_input_fn: str):
    assert mode in ['eval', 'predict']

    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt_dir)
    bert_model = transformers.AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    bert_collator = transformers.DataCollatorWithPadding(tokenizer=bert_tokenizer, max_length=128)
    trainer = transformers.Trainer(model=bert_model, data_collator=bert_collator, tokenizer=bert_tokenizer,
                                   compute_metrics=compute_metrics)

    if mode == 'eval':
        inference_dataset = BinaryDataset(inference_fn, use_label=True, tokenizer=bert_tokenizer)
    elif mode == 'predict':
        inference_dataset = BinaryDataset(inference_fn, use_label=False, tokenizer=bert_tokenizer)
    else:
        raise AssertionError

    inference_results = trainer.predict(inference_dataset)

    reformatted_results = []
    assert len(inference_dataset) == len(inference_results.predictions)

    for ent, scrs in zip(inference_dataset, inference_results.predictions):
        scrs = torch.softmax(torch.tensor(scrs), dim=-1)
        scrs = 0 if scrs[0] > scrs[1] else 1
        reformatted_results.append((ent['sent'], scrs))
    # o = torch.softmax(torch.tensor(o), dim=-1)
    # o = 0 if o[0] > o[1] else 1
    print(f"Inference results: ")
    print(inference_results.metrics)
    with open(result_fn, 'w', encoding='utf8') as ofp:
        json.dump(reformatted_results, ofp, ensure_ascii=False, indent=4)

    if subm_ofn is not None:
        print(f"Writing output json files for submission for ReRanking!")
        subm_gold_lines = read_lm_kbc_jsonl(orig_input_fn)
        subm_perentry_lines = {}

        for ent, scrs in zip(inference_dataset, inference_results.predictions):
            entry_ident_str = f"{ent['SubjectEntity']}#{ent['Relation']}"
            if entry_ident_str not in subm_perentry_lines:
                subm_perentry_lines[entry_ident_str] = {'SubjectEntity': ent['SubjectEntity'],
                                                        'Relation': ent['Relation'],
                                                        'Prompt': [],
                                                        'ObjectEntities': [],
                                                        'Scores': [],}
            else:
                pass
            # If the score for the object being a positive is higher than the score for the object being a negative,
            # add the object into the list
            if scrs[1] > scrs[0]:
                subm_perentry_lines[entry_ident_str]['ObjectEntities'].append(ent['ObjectEntity'])
                subm_perentry_lines[entry_ident_str]['Scores'].append(float(scrs[1]))
            else:
                pass

        subm_perentry_lines = list(subm_perentry_lines.values())
        # print(f"subm_perentry_lines: ")
        # print(subm_perentry_lines)
        scores_per_sr_pair = evaluate_per_sr_pair(subm_perentry_lines, subm_gold_lines, beta=1.0)  # scores per subject-relation pair
        scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)
        scores_per_relation["***Average***"] = {
            "p": sum([x["p"] for x in scores_per_relation.values()]) / len(
                scores_per_relation),
            "r": sum([x["r"] for x in scores_per_relation.values()]) / len(
                scores_per_relation),
            "f1": sum([x["f1"] for x in scores_per_relation.values()]) / len(
                scores_per_relation),
        }

        print(pd.DataFrame(scores_per_relation).transpose().round(3))


        with open(subm_ofn, 'w', encoding='utf8') as subm_ofp:
            for out_item in subm_perentry_lines:
                print(out_item)
                curr_out_line = json.dumps(out_item, ensure_ascii=False)
                subm_ofp.write(curr_out_line+'\n')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--data_mode', type=str, default='development')
    parser.add_argument('--ckpt_dir', type=str, default='../lmkbc_checkpoints/%s_checkpoints_%s%s%s')
    parser.add_argument('--logging_dir', type=str, default='./%s_%s%s%s_tensorboard_logs')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--data_suffix', type=str, default='', help='data_suffix, should start with underscore _')
    parser.add_argument('--model_suffix', type=str, default='', help='model_suffix, should start with underscore _')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta for calculating F-beta scores')

    parser.add_argument('--eval_subset', type=str, default='dev')
    parser.add_argument('--inference_result_fn', type=str, default='./data/%s_%s_binary_results_%s%s.json')

    # flags for collecting data for reranking
    parser.add_argument('--top_k', type=int, default=100)
    # parser.add_argument('--collect_data_gpu_id', type=int, default=-1)
    parser.add_argument('--thresholds_suffix', type=str, default='trial_1.2_blc_withsoftmax')
    parser.add_argument('--hlt_ratio', type=float, default=0.5)
    parser.add_argument('--no_softmax', action='store_true')
    parser.add_argument('--prompt_style', type=str, default=None)
    parser.add_argument('--use_sticky_thres', action='store_true')
    parser.add_argument('--isalive_res_fn', type=str, default=None)
    parser.add_argument('--isindep_res_fn', type=str, default=None)
    parser.add_argument('--tuned_model_name', type=str, default=None)
    parser.add_argument('--thres_beta', type=float, default=1.0)

    args = parser.parse_args()

    args.ckpt_dir = args.ckpt_dir % (args.task_name, args.data_mode, args.data_suffix, args.model_suffix)
    args.logging_dir = args.logging_dir % (args.task_name, args.data_mode, args.data_suffix, args.model_suffix)
    args.inference_result_fn = args.inference_result_fn % (args.task_name, args.eval_subset, args.data_suffix, args.model_suffix)
    assert args.data_mode in ['development', 'submission']
    assert args.job_name in ['collect_data', 'train', 'eval', 'predict']
    assert args.task_name in ['isalive', 'isindep', 'rerank']

    if args.job_name == 'collect_data':
        if args.task_name == 'isalive':
            collect_data_isalive()
        elif args.task_name == 'isindep':
            collect_data_isindep()
        # elif args.task_name == 'locationcategory':
        #     num_labels = 5
        #     raise NotImplementedError
        elif args.task_name == 'rerank':
            thresholds_fn = f"./thresholds/thres_{args.thresholds_suffix}_F%.1f.json" % args.thres_beta
            with open(thresholds_fn, 'r', encoding='utf8') as tfp:
                item = json.load(tfp)
                curr_thresholds = item['thresholds']
                if args.use_sticky_thres is True:
                    curr_sticky_ratios = item['sticky_ratios']
                else:
                    curr_sticky_ratios = {res: None for res in curr_thresholds}

            args.data_suffix += f"_hlt{args.hlt_ratio}_F%.1f" % args.thres_beta

            collect_data_rerank(model_names=[args.model_name, args.tuned_model_name],
                                model_rel_mapping=MODELIDS_REL_MAPPING, top_k=args.top_k,
                                curr_thresholds=curr_thresholds, curr_sticky_ratios=curr_sticky_ratios,
                                hlt_ratio=args.hlt_ratio, suffix=args.data_suffix, use_softmax=(not args.no_softmax),
                                prompt_style=args.prompt_style, tagger_isalive_fn=args.isalive_res_fn,
                                tagger_isindep_fn=args.isindep_res_fn, mask_token='[MASK]')
        else:
            raise NotImplementedError
    elif args.job_name == 'train':
        if args.data_mode == 'development':
            train_fn = os.path.join(args.data_dir, f'train2_{args.task_name}{args.data_suffix}.jsonl')
            dev_fn = os.path.join(args.data_dir, f'dev2_{args.task_name}{args.data_suffix}.jsonl')
            test_fn = os.path.join(args.data_dir, f"dev_{args.task_name}{args.data_suffix}.jsonl")
        elif args.data_mode == 'submission':
            train_fn = os.path.join(args.data_dir, f'train_{args.task_name}{args.data_suffix}.jsonl')
            dev_fn = os.path.join(args.data_dir, f'dev_{args.task_name}{args.data_suffix}.jsonl')
            test_fn = os.path.join(args.data_dir, f"test_{args.task_name}{args.data_suffix}.jsonl")
        else:
            raise AssertionError
        train(args.model_name, train_fn, dev_fn, test_fn, args.ckpt_dir, args.logging_dir, args.lr, args.num_epochs,
              args.no_cuda)
    elif args.job_name in ['eval', 'predict']:
        inference_fn = os.path.join(args.data_dir, f'{args.eval_subset}_{args.task_name}{args.data_suffix}.jsonl')
        if args.task_name == 'rerank':
            subm_fn = f"./outputs/{args.data_suffix}_reranked_{args.eval_subset}.jsonl"
            orig_input_fn = f"./data/{args.eval_subset}.jsonl"
        else:
            subm_fn = None
            orig_input_fn = None
        inference(inference_fn, ckpt_dir=args.model_name, mode=args.job_name, result_fn=args.inference_result_fn,
                  subm_ofn=subm_fn, orig_input_fn=orig_input_fn)
    else:
        raise AssertionError

