import json
import os
import random
from trial_1_2 import RELATIONS

files = os.listdir('./additional_data')
files.sort()
out_fn = './data/silver_train%s.jsonl'

out_entries_per_rel = {rel: [] for rel in RELATIONS}
subj2obj_per_rel = {rel: {} for rel in RELATIONS}


def filter_str(string: str) -> bool:
    if string[0] == 'Q' and string[1:].isdigit():
        return False
    stoptokens = ['-', '(', ')', ',', '\\']
    for tok in stoptokens:
        if tok in string:
            return False
    return True


for f in files:
    fpath = os.path.join('./additional_data', f)
    if not os.path.isfile(fpath):
        continue
    assert f[-5:] == '.json'
    relname = f[:-5]
    assert relname in RELATIONS

    with open(fpath, 'r', encoding='utf8') as rfp:
        item = json.load(rfp)

        if isinstance(item, list):
            for in_entry in item:
                subj_str = in_entry['subjLabel']
                obj_str = in_entry['objLabel']
                if obj_str in subj_str:
                    continue
                elif filter_str(subj_str) is False:
                    continue
                elif filter_str(obj_str) is False:
                    continue
                if subj_str not in subj2obj_per_rel[relname]:
                    subj2obj_per_rel[relname][subj_str] = []
                subj2obj_per_rel[relname][subj_str].append(obj_str)
        else:
            item_head = item['head']
            item_results = item['results']['bindings']
            for entry in item_results:
                # subj = entry['subj']
                # obj = entry['obj']
                subjLabel = entry['subjLabel']
                objLabel = entry['objLabel']
                subj_str = subjLabel['value']
                obj_str = objLabel['value']
                if obj_str in subj_str:
                    continue
                elif filter_str(subj_str) is False:
                    continue
                elif filter_str(obj_str) is False:
                    continue
                if subj_str not in subj2obj_per_rel[relname]:
                    subj2obj_per_rel[relname][subj_str] = []
                subj2obj_per_rel[relname][subj_str].append(obj_str)

    for subj in subj2obj_per_rel[relname]:
        out_entry = {
            'SubjectEntity': subj,
            'Relation': relname,
            'ObjectEntities': [[obj] for obj in subj2obj_per_rel[relname][subj]]
        }
        out_entries_per_rel[relname].append(out_entry)

with open(out_fn % '', 'w', encoding='utf8') as ofp:
    for rel in sorted(RELATIONS):
        print(f"Gathered {len(out_entries_per_rel[rel])} additional entries for the relation # {rel} #;")
        for entry in out_entries_per_rel[rel]:
            out_line = json.dumps(entry, ensure_ascii=False)
            ofp.write(out_line+'\n')

for sample_size in [100, 200, 500, 1000, 2000]:
    with open(out_fn % str(sample_size), 'w', encoding='utf8') as ofp:
        for rel in sorted(RELATIONS):
            curr_sample = out_entries_per_rel[rel] if len(out_entries_per_rel[rel]) < sample_size \
                else random.sample(out_entries_per_rel[rel], k=sample_size)
            for entry in curr_sample:
                out_line = json.dumps(entry, ensure_ascii=False)
                ofp.write(out_line+'\n')

print(f"Finished!")
