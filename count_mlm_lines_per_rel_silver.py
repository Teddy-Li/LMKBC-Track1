import json

num_lines_per_rel = {}

with open('./data/silver_train_mlm_trial_1.2_cmbmined_blc_joint2_withsoftmax_whatcmb.jsonl', 'r', encoding='utf8') as fp:
    for lidx, line in enumerate(fp):
        if lidx % 2000 == 0:
            print(lidx)
        item = json.loads(line)
        cur_item_num = sum(len(x) for x in item['posi_labels'])
        if item['Relation'] not in num_lines_per_rel:
            num_lines_per_rel[item['Relation']] = 0
        num_lines_per_rel[item['Relation']] += cur_item_num

print(f"Final num of sentences: ")
for rel in num_lines_per_rel:
    print(f"Relation: {rel}; number of lines: {num_lines_per_rel[rel]};")

print(f"Finished!")
