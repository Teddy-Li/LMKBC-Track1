import json


for subset in ['train', 'dev']:
    num_objs = {}
    entry_cnts = {}
    with open(f'./data/{subset}.jsonl', 'r', encoding='utf8') as rfp:
        for line in rfp:
            item = json.loads(line)
            if item['Relation'] not in num_objs:
                num_objs[item['Relation']] = []
            num_objs[item['Relation']].append(len(item['ObjectEntities']))
            if item['Relation'] not in entry_cnts:
                entry_cnts[item['Relation']] = 0
            entry_cnts[item['Relation']] += 1

    assert all(len(num_objs[rel]) > 0 for rel in num_objs)
    avg_num_objs = {rel: sum(num_objs[rel])/len(num_objs[rel]) for rel in num_objs}
    total_avg_num_objs = sum(sum(num_objs[rel]) for rel in num_objs) / sum(len(num_objs[rel]) for rel in num_objs)
    print(f"{subset}")
    for rel in avg_num_objs:
        print(f"| {rel} | {avg_num_objs[rel]} |")
    print(f"| Total | {total_avg_num_objs} |")
    print(f"")
    print(f"")

    for rel in entry_cnts:
        print(f"rel: {rel}; entry cnt: {entry_cnts[rel]}")