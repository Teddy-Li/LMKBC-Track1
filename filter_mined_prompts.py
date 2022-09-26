import json
import os
from trial_1_2 import PromptCreator

def lexic_filter(prompt):
    stopwords = [',', ';', '.', '?', '!', ':', '"', "'", '&', '*', '(', ')', '[', ']', '/', '<', '>', '@', '#', '$',
                 '%', '^', '-', '_', 'and', 'or', 'in', 'to', 'of', 'for', 'as', 'the', 'a']
    prompt = prompt.replace('%s', '')
    prompt_lst = prompt.split(' ')
    prompt_lst = [x for x in prompt_lst if len(x) > 0]
    if all(x in stopwords for x in prompt_lst):
        return False
    elif len(prompt_lst) > 10:
        return False
    else:
        return True


def read_what_prompts(mode: str):
    root_dir = f'../LPAQA-master/prompt/{mode}/'
    files = os.listdir(root_dir)
    files.sort()
    prompts_by_relidx = {}
    for f in files:
        fpath = os.path.join(root_dir, f)
        if not os.path.isfile(fpath):
            continue
        assert f[-6:] == '.jsonl'
        rel_idx = f[:-6]
        curr_prompts = []
        with open(fpath, 'r', encoding='utf8') as rfp:
            for line in rfp:
                item = json.loads(line)
                curr_tplt = item['template']
                x_idx = curr_tplt.index('[X]')
                y_idx = curr_tplt.index('[Y]')
                if x_idx < y_idx:
                    aligned = True
                else:
                    aligned = False
                curr_tplt = curr_tplt.replace('[X]', '%s').replace('[Y]', '%s')
                assert curr_tplt.endswith(' .')
                curr_tplt = curr_tplt[:-2] + '.'
                curr_prompts.append((curr_tplt, aligned, 2.0))
        assert rel_idx not in prompts_by_relidx
        prompts_by_relidx[rel_idx] = curr_prompts
    return prompts_by_relidx


LMKBC_REL_TO_WIKIEDATA = {
    'ChemicalCompoundElement': None,
    'CountryBordersWithCountry': 'P47',
    'CountryOfficialLanguage': 'P37',
    'StateSharesBorderState': 'P47',
    'RiverBasinsCountry': None,
    'PersonLanguage': 'P1412',
    'PersonProfession': 'P106',
    'PersonInstrument': 'P1303',
    'PersonEmployer': 'P108',
    'PersonPlaceOfDeath': 'P20',
    'PersonCauseOfDeath': None,
    'CompanyParentOrganization': None
}


def read_prompts_from_file(prompt_fn, out_fn, silver_prompts, esb: str = 'cmb'):
    what_prompts_by_relidx = read_what_prompts('mine')

    print(f"Processing {prompt_fn}")
    prompt_fp = open(prompt_fn, 'r', encoding='utf8')
    out_fp = open(out_fn, 'w', encoding='utf8')
    prompts_per_relation = {}
    for line in prompt_fp:
        item = json.loads(line)
        if item['Relation'] == 'CompanyParentOrganization':
            print(item)
        if item['Relation'] not in prompts_per_relation:
            prompts_per_relation[item['Relation']] = []

        if not lexic_filter(item['prompt']):
            continue

        assert not item['prompt'].endswith('.')
        prompts_per_relation[item['Relation']].append((item['prompt'] + '.', item['alignment_flag'], item['train_frequency']))

    assert len(prompts_per_relation) == 12
    for key in prompts_per_relation:
        prompts_per_relation[key].sort(key=lambda x: x[2], reverse=True)
        prompts_per_relation[key] = prompts_per_relation[key][:30]

        if LMKBC_REL_TO_WIKIEDATA[key] is not None:
            if esb == 'cmb':
                prompts_per_relation[key] += what_prompts_by_relidx[LMKBC_REL_TO_WIKIEDATA[key]]
            elif esb == 'rpc':
                prompts_per_relation[key] = what_prompts_by_relidx[LMKBC_REL_TO_WIKIEDATA[key]]
            else:
                raise AssertionError

        prompts_per_relation[key] += silver_prompts[key]

        new_ppr = []
        for ent_1 in prompts_per_relation[key]:
            found_flag = False
            for ent_2 in new_ppr:
                if ent_1[0] == ent_2[0]:
                    found_flag = True
                    break
            if found_flag:
                continue
            else:
                new_ppr.append(ent_1)

        new_ppr.sort(key=lambda x: x[0])
        prompts_per_relation[key] = new_ppr

        print(f"relation: {key}; number of prompts: {len(prompts_per_relation[key])};")

    json.dump(prompts_per_relation, out_fp, ensure_ascii=False, indent=4)
    prompt_fp.close()
    out_fp.close()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_fn', type=str, default='./prompts/dependency_based_prompts.jsonl')
    # args = parser.parse_args()
    prompt_creator = PromptCreator(fixed_prompt_mode='trial')
    esb = 'rpc'  # 'rpc'

    for input_name in ['dependency_based', 'middle_word', 'combined_mined']:
        input_fn = f'./prompts/{input_name}_prompts.jsonl'
        assert input_fn[-6:] == '.jsonl'
        filtered_fn = input_fn[:-6] + f'_filtered_{esb}.jsonl'
        read_prompts_from_file(input_fn, filtered_fn, prompt_creator.prompt_templates, esb=esb)
