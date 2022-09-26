import argparse
import json
import logging
import random
import copy
import pandas as pd
import torch
import os

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

from evaluate import evaluate_per_sr_pair, combine_scores_per_relation, clean_object

from file_io import read_lm_kbc_jsonl

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

RELATIONS = [
    "CountryBordersWithCountry",
    "CountryOfficialLanguage",
    "StateSharesBorderState",
    "RiverBasinsCountry",
    "ChemicalCompoundElement",
    "PersonLanguage",
    "PersonProfession",
    "PersonInstrument",
    "PersonEmployer",
    "PersonPlaceOfDeath",
    "PersonCauseOfDeath",
    "CompanyParentOrganization",
]

SUBJ_PLACEHOLDERS = {
    "PersonPlaceOfDeath": 'person',
    "PersonCauseOfDeath": 'person',
    "CompanyParentOrganization": 'company',
}

MODELIDS_REL_MAPPING = {
    'ChemicalCompoundElement': 1,
    'CompanyParentOrganization': 0,
    'CountryBordersWithCountry': 0,
    'CountryOfficialLanguage': 0,
    'PersonCauseOfDeath': 0,
    'PersonEmployer': 0,
    'PersonInstrument': 1,
    'PersonLanguage': 1,
    'PersonPlaceOfDeath': 0,
    'PersonProfession': 1,
    'RiverBasinsCountry': 0,
    'StateSharesBorderState': 0
}


class PromptSet(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index) -> T_co:
        return self.prompts[index]


def normalize_to_1(next_layer_res, precon_key_prior):
    if precon_key_prior is not None and all(x > 0 for x in precon_key_prior.values()):
        assert len(next_layer_res) == len(precon_key_prior) == 2
        for k in next_layer_res:
            if k == 'yes':
                next_layer_res[k] *= 2
            next_layer_res[k] /= precon_key_prior[k]
    else:
        pass
    res_sum = sum(next_layer_res.values())
    next_layer_res = {k: x / res_sum for k, x in next_layer_res.items()}
    return next_layer_res


def filter_objs(output, threshold, rel_name, sticky_ratio: float = None):
    assert sticky_ratio is None or 0 < sticky_ratio <= 1
    banned_words = ['I', 'you', 'he', 'she', 'they', 'it', 'we', 'me', 'him', 'her', 'us', 'them', 'mine', 'yours',
                    'his', 'hers', 'ours', 'theirs', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves',
                    'yourselves', 'themselves',
                    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'many', 'some', 'much', 'most', 'any',
                    'home']

    all_elements = ['hydrogen', 'helium', 'lithium', 'beryllium', 'boron', 'carbon', 'nitrogen', 'oxygen', 'fluorine', 'neon',
                    'sodium', 'magnesium', 'aluminium', 'silicon', 'phosphorus', 'sulfur', 'chlorine', 'argon', 'potassium', 'calcium',
                    'scandium', 'titanium', 'vanadium', 'chromium', 'manganese', 'iron', 'cobalt', 'nickel', 'copper', 'zinc',
                    'gallium', 'germanium', 'arsenic', 'selenium', 'bromine', 'krypton', 'rubidium', 'strontium', 'yttrium', 'zirconium',
                    'niobium', 'molybdenum', 'technetium', 'ruthenium', 'rhodium', 'palladium', 'silver', 'cadmium', 'indium', 'tin',
                    'antimony', 'tellurium', 'iodine', 'xenon', 'caesium', 'barium', 'lanthanum', 'cerium', 'praseodymium', 'neodymium',
                    'promethium', 'samarium', 'europium', 'gadolinium', 'terbium', 'dysprosium', 'holmium', 'erbium', 'thulium', 'ytterbium',
                    'lutetium', 'hafnium', 'tantalum', 'tungsten', 'rhenium', 'osmium', 'iridium', 'platinum', 'gold', 'mercury',
                    'thallium', 'lead', 'bismuth', 'polonium', 'astatine', 'radon', 'francium', 'radium', 'actinium', 'thorium',
                    'protactinium', 'uranium', 'neptunium', 'plutonium', 'americium', 'curium', 'berkelium', 'californium', 'einsteinium', 'fermium',
                    'mendelevium', 'nobelium', 'lawrencium', 'rutherfordium', 'dubnium', 'seaborgium', 'bohrium', 'hassium', 'meitnerium', 'darmstadtium',
                    'roentgenium', 'copernicium', 'nihonium', 'flerovium', 'moscovium', 'livermorium', 'tennessine', 'oganesson']

    curr_objs, curr_scores = [], []
    output = {k: v for (k, v) in sorted(output.items(), key=lambda x: x[1], reverse=True)}

    last_scr = None
    for key in output:
        if output[key] < threshold:
            # If the current score is below the threshold, but above a ratio (for instance 80%) of the last score,
            # then still take it, thus ``sticky''; this process carries on until there is one answer less than
            # (for instance 80%) of its last answer.
            if sticky_ratio is not None and last_scr is not None and output[key] / threshold > sticky_ratio:
            # if sticky_ratio is not None and last_scr is not None and output[key] / last_scr > sticky_ratio:
                curr_objs.append(key)
                curr_scores.append(output[key])
                last_scr = output[key]
            else:
                continue
        elif key in banned_words:
            continue
        elif '##' in key:
            continue
        elif rel_name == 'ChemicalCompoundElement' and key not in all_elements:
            # the outputted object must be an element
            continue
        else:
            curr_objs.append(key)
            curr_scores.append(output[key])
            last_scr = output[key]
    return curr_objs, curr_scores


# TODO: Iterative prompts
class PromptCreator:
    def __init__(self, prompt_fn: str = None, fixed_prompt_mode: str = None):
        if prompt_fn is not None:
            print(f"Building PromptCreator from mined prompts in {prompt_fn}!")
            with open(prompt_fn, 'r', encoding='utf8') as rfp:
                self.prompt_templates = json.load(rfp)
            for rel in RELATIONS:
                assert rel in self.prompt_templates
        elif fixed_prompt_mode == 'trial':
            print(f"Building PromptCreator from fixed prompts in # trial # mode!")
            self.prompt_templates = {
                "CountryBordersWithCountry": [("%s shares border with %s.", True, 1.0)],  # (template, aligned_flag, weight)
                "CountryOfficialLanguage": [("The official language of %s is %s.", True, 1.0)],
                "StateSharesBorderState": [("%s shares border with %s.", True, 1.0)],
                "RiverBasinsCountry": [("%s river basins in %s.", True, 1.0)],
                "ChemicalCompoundElement": [("%s consists of %s, which is an element.", True, 1.0)],
                "PersonLanguage": [("%s speaks in %s.", True, 1.0)],
                "PersonProfession": [("%s is a %s by profession", True, 1.0)],
                "PersonInstrument": [("The musician %s plays %s, which is an instrument.", True, 1.0)],
                "PersonEmployer": [("%s works at %s.", True, 1.0)],
                "PersonPlaceOfDeath": [("%s died at %s.", True, 1.0)],
                "PersonCauseOfDeath": [("%s died due to %s.", True, 1.0)],
                "CompanyParentOrganization": [("The parent organization of the company %s is the company %s.", True, 1.0)]
            }
        elif fixed_prompt_mode == 'baseline':
            print(f"Building PromptCreator from fixed prompts in # baseline # mode!")
            self.prompt_templates = {
                "CountryBordersWithCountry": [("%s shares border with %s.", True, 1.0)],
                # (template, aligned_flag, weight)
                "CountryOfficialLanguage": [("The official language of %s is %s.", True, 1.0)],
                "StateSharesBorderState": [("%s shares border with %s.", True, 1.0)],
                "RiverBasinsCountry": [("%s river basins in %s.", True, 1.0)],
                "ChemicalCompoundElement": [("%s consists of %s, which is an element.", True, 1.0)],
                "PersonLanguage": [("%s speaks in %s.", True, 1.0)],
                "PersonProfession": [("%s is a %s by profession", True, 1.0)],
                "PersonInstrument": [("%s plays %s, which is an instrument.", True, 1.0)],
                "PersonEmployer": [("%s is an employer at %s, which is a company.", True, 1.0)],
                "PersonPlaceOfDeath": [("%s died at %s.", True, 1.0)],
                "PersonCauseOfDeath": [("%s died due to %s.", True, 1.0)],
                "CompanyParentOrganization": [
                    ("The parent organization of %s is %s.", True, 1.0)]
            }
        else:
            raise AssertionError

        self.warning_message_flag = False

    def get_num_prompts_per_rel(self):
        nums_per_rel = {}
        for rel in self.prompt_templates:
            nums_per_rel[rel] = len(self.prompt_templates[rel])
        return nums_per_rel

    def create_prompt(self, subject_entity, relation, mask_token):
        # depending on the relation, we fix the prompt
        # prompts: [step1, step2, ...]
        # step: {"precondition1": [prompt1, prompt2, ...], "precondition2": [prompt3, prompt4], ...}
        if relation in ["CountryBordersWithCountry", "CountryOfficialLanguage", "RiverBasinsCountry",
                        "ChemicalCompoundElement", "PersonLanguage", "PersonProfession", "PersonInstrument",
                        "PersonEmployer", "PersonPlaceOfDeath", "PersonCauseOfDeath", "CompanyParentOrganization"]:
            laststep_prompts = []
            for tplt in self.prompt_templates[relation]:
                if tplt[1] is True:
                    laststep_prompts.append((tplt[0] % (subject_entity, mask_token), tplt[2]))
                elif tplt[1] is False:
                    laststep_prompts.append((tplt[0] % (mask_token, subject_entity), tplt[2]))
                else:
                    raise AssertionError

            if relation in ["PersonPlaceOfDeath", "PersonCauseOfDeath"]:
                # prompts = [{"": laststep_prompts}]
                # if not self.warning_message_flag:
                #     print(f"Binary classifier preconditions not used for PersonPlaceOfDeath and PersonCauseOfDeath!")
                #     self.warning_message_flag = True
                prompts = [{"": [(f"Has {subject_entity} died? {mask_token}.", 1.0)]}, {"yes": laststep_prompts, "no": None}]
            elif relation in ["CompanyParentOrganization"]:
                prompts = [{"": [(f"Does an organization own {subject_entity}? {mask_token}.", 1.0)]}, {"yes": laststep_prompts, "no": None}]
            else:
                prompts = [{"": laststep_prompts}]
        elif relation == "StateSharesBorderState":
            prompts = [{    "": [(f"{subject_entity}, as a place, is a {mask_token}.", 1.0)]
                        },
                       {    "state": [],
                            "province": [],
                            "region": [],
                            "department": [],
                            "city": []
                        }
                       ]
            assert len(prompts) == 2
            for tplt in self.prompt_templates[relation]:
                subject_ent_forms_per_key = {
                    "state": f"{subject_entity} state",
                    "province": f"{subject_entity} province",
                    "region": f"{subject_entity} region",
                    "department": f"{subject_entity} department",
                    "city": f"The city of {subject_entity}"
                }
                if tplt[1] is True:
                    for key in prompts[1]:
                        prompts[1][key].append((tplt[0] % (subject_ent_forms_per_key[key], mask_token), tplt[2]))
                elif tplt[1] is False:
                    for key in prompts[1]:
                        prompts[1][key].append((tplt[0] % (mask_token, subject_ent_forms_per_key[key]), tplt[2]))
                else:
                    raise AssertionError
        else:
            raise AssertionError
        return prompts


def gather_key_priors(pipe, model_rel_mapping, prompt_creator: PromptCreator, mask_token: str):
    result = {}

    for relation in SUBJ_PLACEHOLDERS:
        prompts = prompt_creator.create_prompt(SUBJ_PLACEHOLDERS[relation], relation, mask_token)
        assert len(prompts) <= 2
        prec_prompts = prompts[0][""]
        prec_keys = list(prompts[1].keys())

        prec_all_res = {}
        prec_key_res = {}

        for p, _ in prec_prompts:
            curr_res = pipe(p)
            for ent in curr_res:
                if ent['token_str'] not in prec_all_res:
                    prec_all_res[ent['token_str']] = ent['score']
                prec_all_res[ent['token_str']] = max(prec_all_res[ent['token_str']], ent['score'])

        for key in prec_keys:
            if key in prec_all_res:
                prec_key_res[key] = prec_all_res[key]
            else:
                prec_key_res[key] = 0

        result[relation] = prec_key_res
    return result


def perentry_prompting(entry, pipe, mask_token, key_priors: dict, prompt_creator: PromptCreator,
                       tagger_isalive_val: int = None, tagger_isindep_val: int = None):
    entry_ps = prompt_creator.create_prompt(subject_entity=entry['SubjectEntity'],
                                          relation=entry['Relation'],
                                          mask_token=mask_token)

    subject_entity_words = entry['SubjectEntity'].split()
    if len(entry_ps) > 2:
        raise NotImplementedError

    # if entry['Relation'] == 'PersonPlaceOfDeath':
    #     print(f"!")

    joined_outputs_per_prompt = None
    final_prompts = None
    precondition = ""
    for lidx, layer in enumerate(entry_ps):
        # In the following conditions, pre-conditions from previous layers can be overwritten by (pseudo-)oracle
        overwrite_precondition = None
        for key in layer:
            # TODO: check which way is better: keep or remove redundant type labels
            # if len(key) > 0 and key in [x.lower() for x in subject_entity_words]:
            if len(key) > 0 and key in subject_entity_words:
                if overwrite_precondition is not None:
                    print(f"Multiple keywords! {key}; {overwrite_precondition}")
                overwrite_precondition = key

        if entry['Relation'] in ['PersonPlaceOfDeath', 'PersonCauseOfDeath'] and tagger_isalive_val is not None:
            if tagger_isalive_val == 1:
                overwrite_precondition = 'no' if 'no' in layer else None  # the classifier is for isalive but the prompts are for isdead!
            elif tagger_isalive_val == 0:
                overwrite_precondition = 'yes' if 'yes' in layer else None
            else:
                raise AssertionError
        if entry['Relation'] in ['CompanyParentOrganization'] and tagger_isindep_val is not None:
            if tagger_isindep_val == 1:
                overwrite_precondition = 'no' if 'no' in layer else None  # the classifier is for isindep but the prompts are for issubcompany!
            elif tagger_isindep_val == 0:
                overwrite_precondition = 'yes' if 'yes' in layer else None
            else:
                raise AssertionError
        # if entry['Relation'] in ['PersonPlaceOfDeath', 'PersonCauseOfDeath', 'CompanyParentOrganization'] and \
        #     always_predict is True and 'yes' in layer:
        #     overwrite_precondition = 'yes'

        if overwrite_precondition is not None:
            precondition = overwrite_precondition

        layer_ps = layer[precondition]
        final_prompts = layer_ps
        if final_prompts is None:
            assert precondition == 'no'
            final_prompts = layer['yes']

        if layer_ps is None:
            precondition = ""
            joined_outputs_per_prompt = []
            continue
        else:
            joined_outputs_per_prompt = [{} for x in layer_ps]
            for pidx, (p, _) in enumerate(layer_ps):
                probe_outputs = pipe(p)
                for ent in probe_outputs:
                    assert ent['token_str'] not in joined_outputs_per_prompt[pidx]
                    joined_outputs_per_prompt[pidx][ent['token_str']] = ent['score']

        if lidx == len(entry_ps) - 1:
            continue
        else:
            next_layer_keys = list(entry_ps[lidx + 1].keys())
            next_layer_res = {}
            assert len(joined_outputs_per_prompt) == 1  # if this is the precondition layer, then there should be only one prompt.
            joined_outputs_per_prompt = joined_outputs_per_prompt[0]
            for key in next_layer_keys:
                next_layer_res[key] = 0
                if key in joined_outputs_per_prompt:
                    next_layer_res[key] = joined_outputs_per_prompt[key]

            if sum(next_layer_res.values()) == 0:
                precondition = random.choice(next_layer_keys)
            else:
                next_layer_res = normalize_to_1(next_layer_res, key_priors[entry['Relation']] if entry[
                                                                                                     'Relation'] in key_priors else None)
                precondition = sorted(next_layer_res.items(), key=lambda x: x[1], reverse=True)[0][
                    0]  # the key of the first entry in the reverse-sorted dict by value.
            continue

    for pidx, ppjo in enumerate(joined_outputs_per_prompt):
        scrs_orig_sum = sum(ppjo.values())
        objs, scrs = filter_objs(ppjo, 0, entry['Relation'], sticky_ratio=None)
        if scrs_orig_sum > 0 and len(objs) > 0:
            scrs_new_sum = sum(scrs)
            scaleup_ratio = scrs_orig_sum / scrs_new_sum
            scrs = [s*scaleup_ratio for s in scrs]
            assert len(objs) == len(scrs)
            joined_outputs_per_prompt[pidx] = {o: s for o, s in zip(objs, scrs)}
        else:
            pass
    raw_results = {'SubjectEntity': entry['SubjectEntity'], 'Relation': entry['Relation'], 'Results': joined_outputs_per_prompt}
    return raw_results, final_prompts


def run(model_names: dict, model_rel_mapping: dict, input_fn: str, top_k: int, thresholds: dict, sticky_ratios: dict,
        gpu_id: int, use_softmax: bool, prompt_creator: PromptCreator, tagger_isalive_res_fn: str = None,
        tagger_isindep_res_fn: str = None):

    logger.info(f"Loading the model \"{model_names['default']}\"...")

    dft_tokenizer = AutoTokenizer.from_pretrained(model_names['default'])
    dft_model = AutoModelForMaskedLM.from_pretrained(model_names['default'])
    mask_token = dft_tokenizer.mask_token
    default_pipe = pipeline(
        task="fill-mask",
        model=dft_model,
        tokenizer=dft_tokenizer,
        top_k=top_k,
        device=gpu_id if torch.cuda.is_available() else -1,
        use_softmax=use_softmax
    )

    # gather_key_priors is not affected by the prompts in the last layer.
    key_priors = gather_key_priors(default_pipe, model_rel_mapping, prompt_creator, mask_token)
    del default_pipe
    del dft_model
    del dft_tokenizer

    if tagger_isalive_res_fn is not None:
        with open(tagger_isalive_res_fn, 'r', encoding='utf8') as fp:
            tagger_isalive_results = json.load(fp)
    else:
        tagger_isalive_results = None
    isalive_cnt = 0

    if tagger_isindep_res_fn is not None:
        with open(tagger_isindep_res_fn, 'r', encoding='utf8') as fp:
            tagger_isindep_results = json.load(fp)
    else:
        tagger_isindep_results = None
    isindep_cnt = 0

    # Load the input file
    logger.info(f"Loading the input file \"{input_fn}\"...")
    input_rows = read_lm_kbc_jsonl(input_fn)
    logger.info(f"Loaded {len(input_rows):,} rows.")

    # Run the model
    logger.info(f"Running the model...")
    all_outputs = [None for x in input_rows]
    all_used_prompts = [None for x in input_rows]

    for rel in RELATIONS:
        print(f"Processing for relation {rel}......")
        model_name = model_names[rel]
        if model_name is None:
            model_name = model_names['default']
        logger.info(f"Loading the model \"{model_name}\"...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        assert mask_token == tokenizer.mask_token

        pipe = pipeline(
            task="fill-mask",
            model=model,
            tokenizer=tokenizer,
            top_k=top_k,
            device=gpu_id if torch.cuda.is_available() else -1,
            use_softmax=use_softmax
        )

        for eidx, entry in enumerate(input_rows):
            # if eidx % 200 == 0:
            #     print(f"entry idx: {eidx}")
            if entry['Relation'] != rel:
                continue

            tagger_isalive_val = None
            tagger_isindep_val = None
            if tagger_isalive_results is not None and entry['Relation'] in ['PersonPlaceOfDeath', 'PersonCauseOfDeath']:
                print(f"entry subject entity: {entry['SubjectEntity']}; isalive tagger results prompt: {tagger_isalive_results[isalive_cnt]}; should be aligned!")
                tagger_isalive_val = tagger_isalive_results[isalive_cnt][1]
                isalive_cnt += 1
            if tagger_isindep_results is not None and entry['Relation'] in ['CompanyParentOrganization']:
                print(f"entry subject entity: {entry['SubjectEntity']}; isindep tagger results prompt: {tagger_isindep_results[isindep_cnt]}; should be aligned!")
                tagger_isindep_val = tagger_isindep_results[isindep_cnt][1]
                isindep_cnt += 1

            joined_outputs_per_prompt, final_prompts = perentry_prompting(entry, pipe, mask_token, key_priors, prompt_creator,
                                                               tagger_isalive_val=tagger_isalive_val,
                                                               tagger_isindep_val=tagger_isindep_val)
            assert all_outputs[eidx] is None and all_used_prompts[eidx] is None
            all_outputs[eidx] = joined_outputs_per_prompt
            all_used_prompts[eidx] = final_prompts

        del pipe
        del model
        del tokenizer

    # results = []
    # for row, prompts, outputs in zip(input_rows, all_used_prompts, all_outputs):
    #     assert len(outputs) == len(prompts)
    #     assembled_objs = {}
    #     for pidx, ppjo in enumerate(outputs):
    #         curr_objs, curr_scores = filter_objs(ppjo, thresholds[row['Relation']][pidx], row['Relation'],
    #                                              sticky_ratio=sticky_ratios[row['Relation']][pidx])
    #         for o, s in zip(curr_objs, curr_scores):
    #             if o not in assembled_objs:
    #                 assembled_objs[o] = 0
    #             assembled_objs[o] += s * prompt_weights_per_rel[row['Relation']][pidx]
    #
    #     result = {
    #         "SubjectEntity": row["SubjectEntity"],
    #         "Relation": row["Relation"],
    #         "Prompt": prompts,
    #         "ObjectEntities": curr_objs,
    #         "Scores": curr_scores
    #     }
    #     results.append(result)

    return all_outputs, all_used_prompts


# The modified version of search_best_thesholds is faster because it avoid re-calculating the scores every time, but it
# is not compatible with the interative sampling idea, because the sampling process may halt at some point.

# Here we search for the best thresholds for each prompt in each relation individually.
def search_best_thresholds(model_names: dict, model_rel_mapping: dict, input_fn: str, top_k: int, gpu_id: int,
                           use_softmax: bool, prompt_creator: PromptCreator, tagger_isalive_res_fn: str,
                           tagger_isindep_res_fn: str, beta: float, search_sticky: bool):  #, exit_criterion: int):
    if use_softmax:
        # init_thres = 0.2
        # init_step = 0.1
        begin = 0.01
        end = 0.95
        step = 0.01
    else:
        begin = -5
        end = 25
        step = 0.1
        # init_thres = 10
        # init_step = 1
    # assert exit_criterion > 0

    gt_rows = read_lm_kbc_jsonl(input_fn)
    raw_outputs, raw_used_prompts = run(model_names, model_rel_mapping, input_fn, top_k, {rel: -10 for rel in RELATIONS},
                      {rel: None for rel in RELATIONS}, gpu_id=gpu_id, use_softmax=use_softmax, prompt_creator=prompt_creator,
                      tagger_isalive_res_fn=tagger_isalive_res_fn, tagger_isindep_res_fn=tagger_isindep_res_fn)

    num_prompts_per_rel = prompt_creator.get_num_prompts_per_rel()

    best_thresholds = {rel: [None] * num_prompts_per_rel[rel] for rel in RELATIONS}
    best_sticky_ratios = {rel: [None] * num_prompts_per_rel[rel] for rel in RELATIONS}
    best_scores = {rel: [None] * num_prompts_per_rel[rel] for rel in RELATIONS}

    if search_sticky:
        sticky_ratio_set = [0.4 + 0.03 * x for x in range(20)] + [None]
    else:
        sticky_ratio_set = [None]
    sticky_ratio_set.reverse()

    curr_t = begin
    tidx = 0

    while curr_t < end:
        if tidx % 10 == 0:
            print(f"tidx: {tidx};")

        # curr_thresholds = {rel: curr_t for rel in RELATIONS}

        for curr_sr in sticky_ratio_set:
            for rel in RELATIONS:
                curr_allentry_results_per_prompt = [[] for x in range(num_prompts_per_rel[rel])]
                for res in raw_outputs:
                    if res['Relation'] != rel:  # process entries with one RELATION at a time, since different relations have different numbers of prompts.
                        continue
                    if len(res['Results']) == 0:
                        res['Results'] = [{} for x in range(num_prompts_per_rel[rel])]
                    assert len(res['Results']) == num_prompts_per_rel[rel], print(f"len results: {len(res['Results'])}; num prompts per rel: {num_prompts_per_rel[rel]};")
                    for pidx, ppoj in enumerate(res['Results']):
                        curr_objs, curr_scores = filter_objs(ppoj, curr_t, res['Relation'], sticky_ratio=curr_sr)
                        curr_ppres = {
                            'SubjectEntity': res['SubjectEntity'],
                            'ObjectEntities': curr_objs,
                            'Relation': res['Relation'],
                            'Scores': curr_scores
                        }
                        try:
                            curr_allentry_results_per_prompt[pidx].append(curr_ppres)
                        except Exception as e:
                            print(e)
                            print(f"res results: ")
                            print(res['Results'])
                            print(f"num_prompts_per_rel[rel]: {num_prompts_per_rel[rel]}")
                            print(f"pidx: {pidx};")
                            print(f"len curr_allXXX: {len(curr_allentry_results_per_prompt)}")
                            print(curr_allentry_results_per_prompt)
                            raise

                for pidx, curr_oneprompt_allentry_results in enumerate(curr_allentry_results_per_prompt):
                    oneprompt_scores_per_sr_pair = evaluate_per_sr_pair(curr_oneprompt_allentry_results, gt_rows, beta=beta)  # scores per subject-relation pair
                    oneprompt_scores_per_relation = combine_scores_per_relation(oneprompt_scores_per_sr_pair)

                    if best_scores[rel][pidx] is None or oneprompt_scores_per_relation[rel]['f1'] > best_scores[rel][pidx]['f1']:
                        best_scores[rel][pidx] = oneprompt_scores_per_relation[rel]
                        best_thresholds[rel][pidx] = curr_t
                        best_sticky_ratios[rel][pidx] = curr_sr

        curr_t += step
        tidx += 1
        continue

    best_scores_idxes = {rel: sorted([(pidx, scr) for pidx, scr in enumerate(best_scores[rel])], key=lambda x: x[1]['f1'], reverse=True) for rel in best_scores}
    maximum_numprompts = min(max([len(best_scores[x]) for x in best_scores]), 10)  # cap the maximum number of prompts used to 10

    assemble_best_numprompts = {rel: None for rel in RELATIONS}
    assemble_best_thresholds = {rel: None for rel in RELATIONS}
    assemble_best_sticky_ratios = {rel: None for rel in RELATIONS}
    assemble_best_scores = {rel: None for rel in RELATIONS}

    assemble_mode = 'avg'

    for asb_size in range(1, maximum_numprompts+1):
        asb_curr_thres = begin
        while asb_curr_thres < end:
            for asb_curr_sticky_ratio in sticky_ratio_set:
                asb_curr_results = []
                for res in raw_outputs:
                    curr_res = {
                        'SubjectEntity': res['SubjectEntity'],
                        'Relation': res['Relation'],
                        'ObjectEntities': [],
                        'Scores': []
                    }
                    assembled_oneentry_joined_outputs = {}
                    # do not filter the individual-prompt scores, filter only the assembled ones.
                    for pidx, _ in best_scores_idxes[res['Relation']][:asb_size]:
                        for obj in res['Results'][pidx]:
                            scr = res['Results'][pidx][obj]
                            if obj not in assembled_oneentry_joined_outputs:
                                assembled_oneentry_joined_outputs[obj] = 0
                            if assemble_mode == 'avg':  # because the set of prompts is the same, avg is the same as sum
                                assembled_oneentry_joined_outputs[obj] += scr
                            elif assemble_mode == 'max':
                                assembled_oneentry_joined_outputs[obj] = max(scr, assembled_oneentry_joined_outputs[obj])
                            else:
                                raise AssertionError
                    asb_curr_objs, asb_curr_scrs = filter_objs(assembled_oneentry_joined_outputs, asb_curr_thres,
                                                                           res['Relation'],
                                                                           sticky_ratio=asb_curr_sticky_ratio)
                    curr_res['ObjectEntities'] = asb_curr_objs
                    curr_res['Scores'] = asb_curr_scrs
                    asb_curr_results.append(curr_res)

                asb_scores_per_sr_pair = evaluate_per_sr_pair(asb_curr_results, gt_rows, beta=beta)  # scores per subject-relation pair
                asb_scores_per_relation = combine_scores_per_relation(asb_scores_per_sr_pair)

                for rel in RELATIONS:
                    # Update: added a 0.01 margin to avoid overfitting.
                    margin = 0.0 if asb_size == assemble_best_numprompts[rel] else 0.01
                    if assemble_best_scores[rel] is None or asb_scores_per_relation[rel]['f1'] > assemble_best_scores[rel]['f1'] + margin:
                        assemble_best_scores[rel] = asb_scores_per_relation[rel]
                        assemble_best_numprompts[rel] = asb_size
                        assemble_best_thresholds[rel] = asb_curr_thres
                        assemble_best_sticky_ratios[rel] = asb_curr_sticky_ratio
            asb_curr_thres += step

        print(f"asb_size: {asb_size}; best scores so far: ")
        print(pd.DataFrame(assemble_best_scores).transpose().round(3))

    best_results = []
    for res in raw_outputs:
        curr_res = {
            'SubjectEntity': res['SubjectEntity'],
            'Relation': res['Relation'],
            'ObjectEntities': [],
            'Scores': []
        }
        assembled_oneentry_joined_outputs = {}
        for pidx, _ in best_scores_idxes[res['Relation']][:assemble_best_numprompts[res['Relation']]]:
            for obj in res['Results'][pidx]:
                scr = res['Results'][pidx][obj]
                if obj not in assembled_oneentry_joined_outputs:
                    assembled_oneentry_joined_outputs[obj] = 0
                if assemble_mode == 'avg':  # because the set of prompts is the same, avg is the same as sum
                    assembled_oneentry_joined_outputs[obj] += scr
                elif assemble_mode == 'max':
                    assembled_oneentry_joined_outputs[obj] = max(scr, assembled_oneentry_joined_outputs[obj])
                else:
                    raise AssertionError

        asb_curr_objs, asb_curr_scores = filter_objs(assembled_oneentry_joined_outputs, assemble_best_thresholds[res['Relation']],
                                             res['Relation'], sticky_ratio=assemble_best_sticky_ratios[res['Relation']])
        curr_res['ObjectEntities'] = asb_curr_objs
        curr_res['Scores'] = asb_curr_scores
        best_results.append(curr_res)

    assert all(assemble_best_thresholds[rel] is not None for rel in assemble_best_thresholds)

    assemble_best_scores["***Average***"] = {
        "p": sum([x["p"] for x in assemble_best_scores.values()]) / len(
            assemble_best_scores),
        "r": sum([x["r"] for x in assemble_best_scores.values()]) / len(
            assemble_best_scores),
        "f1": sum([x["f1"] for x in assemble_best_scores.values()]) / len(
            assemble_best_scores),
    }
    assemble_best_scores = {k: v for k, v in sorted(assemble_best_scores.items(), key=lambda x: x[0])}

    print(pd.DataFrame(assemble_best_scores).transpose().round(3))

    print(f"Final num prompts: ")
    print(assemble_best_numprompts)

    print(f"Final thresholds: ")
    print(assemble_best_thresholds)

    print(f"Final Sticky Ratios: ")
    print(assemble_best_sticky_ratios)

    best_scores_idxes = {rel: [(idx, scr, prompt_creator.prompt_templates[rel][idx]) for (idx, scr) in best_scores_idxes[rel]] for rel in best_scores_idxes}

    return assemble_best_thresholds, assemble_best_sticky_ratios, assemble_best_numprompts, best_results, best_scores_idxes


def run_single_ranking_avg(model_names: dict, model_rel_mapping: dict, input_fn: str, top_k: int, curr_thresholds: dict,
                           curr_sticky_ratios: dict, curr_numprompts, prompt_order_idxes, gpu_id: int, use_softmax: bool,
                           prompt_creator: PromptCreator, tagger_isalive_res_fn: str,
                           tagger_isindep_res_fn: str, relaxed_thres: float = 1.0, beta: float = 1.0):

    assemble_mode = 'avg'
    curr_thresholds = {k: curr_thresholds[k] * relaxed_thres for k in curr_thresholds}
    num_prompts_per_rel = prompt_creator.get_num_prompts_per_rel()
    for rel in RELATIONS:
        assert len(prompt_order_idxes[rel]) == len(prompt_creator.prompt_templates[rel])
        for pidx, _, p in prompt_order_idxes[rel]:
            assert prompt_creator.prompt_templates[rel][pidx][0] == p[0]
    raw_outputs, raw_used_prompts = run(model_names, model_rel_mapping, input_fn, top_k, {rel: -10 for rel in RELATIONS},
                                        {rel: None for rel in RELATIONS}, gpu_id=gpu_id, use_softmax=use_softmax,
                                        prompt_creator=prompt_creator, tagger_isalive_res_fn=tagger_isalive_res_fn,
                                        tagger_isindep_res_fn=tagger_isindep_res_fn)

    assembled_results = []
    for res in raw_outputs:
        curr_res = {
            'SubjectEntity': res['SubjectEntity'],
            'Relation': res['Relation'],
            'ObjectEntities': [],
            'Scores': []
        }
        if len(res['Results']) == 0:
            res['Results'] = [{} for x in range(num_prompts_per_rel[res['Relation']])]

        assembled_oneentry_joined_outputs = {}
        for pidx, _, _ in prompt_order_idxes[res['Relation']][:curr_numprompts[res['Relation']]]:
            # print(len(res['Results']))
            # print(res['Results'])
            # print(prompt_order_idxes[res['Relation']])
            # print(curr_numprompts[res['Relation']])
            # print("")
            for obj in res['Results'][pidx]:
                scr = res['Results'][pidx][obj]
                if obj not in assembled_oneentry_joined_outputs:
                    assembled_oneentry_joined_outputs[obj] = 0
                if assemble_mode == 'avg':  # because the set of prompts is the same, avg is the same as sum
                    assembled_oneentry_joined_outputs[obj] += scr
                elif assemble_mode == 'max':
                    assembled_oneentry_joined_outputs[obj] = max(scr, assembled_oneentry_joined_outputs[obj])
                else:
                    raise AssertionError

        asb_curr_objs, asb_curr_scores = filter_objs(assembled_oneentry_joined_outputs,
                                                     curr_thresholds[res['Relation']],
                                                     res['Relation'],
                                                     sticky_ratio=curr_sticky_ratios[res['Relation']])
        curr_res['ObjectEntities'] = asb_curr_objs
        curr_res['Scores'] = asb_curr_scores
        assembled_results.append(curr_res)

    gt_rows = read_lm_kbc_jsonl(input_fn)
    scores_per_sr_pair = evaluate_per_sr_pair(assembled_results, gt_rows, beta=beta)  # scores per subject-relation pair
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
    return assembled_results


class PromptWeights(torch.nn.Module):
    def __init__(self, num_prompts_per_rel):
        super().__init__()

        # weights are randomly initialized between [-1, 1); aligned with labels: 1 for positive, -1 for negative
        # the +1 is for the threshold term. ; Update: we do not +1, instead, we define the bias term as 1-sum(weights) ; Update: we add back the +1 and use softmax to have weighted average
        self.weights = torch.nn.ParameterDict({rel: torch.nn.Parameter(2*torch.rand(num_prompts_per_rel[rel]+1)-1,
                                                requires_grad=True) for rel in RELATIONS})
        self.softmax = torch.nn.Softmax()
        self.loss = torch.nn.L1Loss()
        print(f"self parameters: ")
        print(list(self.parameters()))

    def forward(self, rel: str, pred_scores: dict, gold_objs: list):
        asb_true_predictions = []
        asb_false_predictions = []
        asb_dict = {}
        for kidx, key in enumerate(pred_scores):
            curr_obj_pred_scores = copy.copy(pred_scores[key])
            curr_obj_pred_scores.append(1.0)
            assert len(curr_obj_pred_scores) == len(self.weights[rel])
            weights = self.softmax(self.weights[rel])
            asb_score = torch.dot(torch.tensor(curr_obj_pred_scores), weights)
            # asb_score += (1 - torch.sum(self.weights[rel]))
            # if asb_score > 0.2:
            asb_dict[key] = asb_score.item()
            for gold_synset in gold_objs:
                assert isinstance(gold_synset, list)
                # here the keys should already have been cleaned! So no need to clean them again!
                if key in gold_synset:
                    asb_true_predictions.append(asb_score)
                else:
                    asb_false_predictions.append(asb_score)
        if len(asb_true_predictions) == 0 and len(asb_false_predictions) == 0:
            return {}, None
        else:
            # asb_predictions = torch.stack(asb_predictions)
            # loss1 = self.loss(asb_predictions, asb_labels)
            # loss2 = (torch.sum(asb_predictions) / asb_predictions.shape[0] + 1.0) / 2.0
            # # print(f"loss1: {loss1}; loss2: {loss2}")
            # loss = 0.7*loss1 + 0.3*loss2
            asb_false_sample_size = round(len(asb_true_predictions)*2+1)
            asb_false_sample = random.sample(asb_false_predictions, k=asb_false_sample_size) if \
                asb_false_sample_size < len(asb_false_predictions) else asb_false_predictions
            asb_predictions = asb_true_predictions + asb_false_sample
            asb_labels = [1.0 for x in asb_true_predictions] + [0.0 for x in asb_false_sample]

            asb_predictions = torch.stack(asb_predictions)
            asb_labels = torch.tensor(asb_labels)
            loss = self.loss(asb_predictions, asb_labels)
            return asb_dict, loss


class WeightsDataset(torch.utils.data.Dataset):
    def __init__(self, raw_outputs, inputs):
        self.raw_outputs = raw_outputs
        self.entries = []
        for res, inp in zip(self.raw_outputs, inputs):
            rel = res['Relation']
            all_prompts_outputs = {}
            # first collect the union set of all predicted objects
            for oneprompt_output in res['Results']:
                for key in oneprompt_output:
                    key_sf = clean_object(key)
                    if key_sf not in all_prompts_outputs:
                        all_prompts_outputs[key_sf] = []
            # then record the scores for each object, if the object is not proposed from the current prompt,
            # record 0.0
            for oneprompt_output in res['Results']:
                cleaned_opo = {clean_object(key): val for (key, val) in oneprompt_output.items()}
                for key in all_prompts_outputs:
                    if key in cleaned_opo:
                        all_prompts_outputs[key].append(cleaned_opo[key])
                    else:
                        all_prompts_outputs[key].append(0.0)
            self.entries.append({'rel': rel, 'dict': all_prompts_outputs, 'gold': inp['ObjectEntities'],
                                 'subj': inp['SubjectEntity']})

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]['rel'], self.entries[idx]['dict'], self.entries[idx]['gold'], self.entries[idx]['subj']


def learn_weights(model_names: dict, model_rel_mapping: dict, train_input_fn: str, dev_input_fn: str, train_cache_fn: str,
                  dev_cache_fn: str, cache_usage_mode: str, top_k: int, gpu_id: int, use_softmax: bool, prompt_creator: PromptCreator,
                  tagger_isalive_res_fn: str, tagger_isindep_res_fn: str, num_epochs: int, lr: float, grad_acc_steps: int,
                  model_out_dir: str):
    num_prompts_per_rel = prompt_creator.get_num_prompts_per_rel()
    prompt_weights_learner = PromptWeights(num_prompts_per_rel)
    optimizer = torch.optim.AdamW([prompt_weights_learner.weights[rel] for rel in prompt_weights_learner.weights], lr=lr, weight_decay=1e-5, )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.965)
    assert cache_usage_mode is None or cache_usage_mode in ['store', 'load']

    if cache_usage_mode is None or cache_usage_mode == 'store':
        train_raw_outputs, train_raw_used_prompts = run(model_names, model_rel_mapping, train_input_fn, top_k, {rel: -10 for rel in RELATIONS},
                                            {rel: None for rel in RELATIONS}, gpu_id=gpu_id, use_softmax=use_softmax,
                                            prompt_creator=prompt_creator, tagger_isalive_res_fn=tagger_isalive_res_fn,
                                            tagger_isindep_res_fn=tagger_isindep_res_fn)

        dev_raw_outputs, dev_raw_used_prompts = run(model_names, model_rel_mapping, dev_input_fn, top_k,
                                                        {rel: -10 for rel in RELATIONS},
                                                        {rel: None for rel in RELATIONS}, gpu_id=gpu_id,
                                                        use_softmax=use_softmax, prompt_creator=prompt_creator,
                                                        tagger_isalive_res_fn=tagger_isalive_res_fn,
                                                        tagger_isindep_res_fn=tagger_isindep_res_fn)
        if cache_usage_mode == 'store':
            with open(train_cache_fn, 'w', encoding='utf8') as ofp:
                for res in train_raw_outputs:
                    out_line = json.dumps(res, ensure_ascii=False)
                    ofp.write(out_line + '\n')

            with open(dev_cache_fn, 'w', encoding='utf8') as ofp:
                for res in dev_raw_outputs:
                    out_line = json.dumps(res, ensure_ascii=False)
                    ofp.write(out_line + '\n')
    elif cache_usage_mode == 'load':
        train_raw_outputs = []
        dev_raw_outputs = []
        with open(train_cache_fn, 'r', encoding='utf8') as rfp:
            for line in rfp:
                item = json.loads(line)
                train_raw_outputs.append(item)
        with open(dev_cache_fn, 'r', encoding='utf8') as rfp:
            for line in rfp:
                item = json.loads(line)
                dev_raw_outputs.append(item)
    else:
        raise AssertionError

    if gpu_id is not None and gpu_id >= 0 and torch.cuda.is_available():
        prompt_weights_learner.to(torch.device(f'cuda:{gpu_id}'))
    train_gold = read_lm_kbc_jsonl(train_input_fn)
    dev_gold = read_lm_kbc_jsonl(dev_input_fn)

    train_dataset = WeightsDataset(train_raw_outputs, train_gold)
    dev_dataset = WeightsDataset(dev_raw_outputs, dev_gold)

    best_f1 = None
    best_scores_per_relation = None

    for epoch in range(num_epochs):
        print(f"Processing epoch number {epoch}; current lr: {scheduler.get_last_lr()}!")
        idxes = [x for x in range(len(train_dataset))]
        random.shuffle(idxes)
        # train for one epoch
        prompt_weights_learner.train()
        optimizer.zero_grad()
        total_train_loss = 0.0

        for eidx in idxes:
            if eidx % grad_acc_steps == 0 and eidx != 0:
                optimizer.step()
                optimizer.zero_grad()

            rel, dct, gld, subj = train_dataset[eidx]
            objs, loss = prompt_weights_learner(rel, dct, gld)
            if loss is not None:
                loss.backward()
                total_train_loss += loss
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        # test on dev set
        with torch.no_grad():
            prompt_weights_learner.eval()

            raw_outputs = []
            total_dev_loss = 0.0
            for eidx, entry in enumerate(dev_dataset):
                rel, dct, gld, subj = entry
                objs, loss = prompt_weights_learner(rel, dct, gld)
                raw_outputs.append({'subj': subj, 'rel': rel, 'objs': objs})
                if loss is not None:
                    total_dev_loss += loss
            print(f"Total train loss: {total_train_loss}; Total dev loss: {total_dev_loss};")

            begin = 0.01
            end = 0.95
            step = 0.01

            curr_best_thresholds = {rel: None for rel in RELATIONS}
            curr_best_scores = {rel: None for rel in RELATIONS}

            curr_t = begin
            while curr_t < end:
                curr_outputs = []
                for curr_raw in raw_outputs:
                    objs, scrs = filter_objs(curr_raw['objs'], curr_t, curr_raw['rel'], None)
                    curr_single_o = {
                        'SubjectEntity': curr_raw['subj'],
                        'Relation': curr_raw['rel'],
                        'ObjectEntities': objs,
                        'Scores': scrs
                    }
                    curr_outputs.append(curr_single_o)
                scores_per_sr_pair = evaluate_per_sr_pair(curr_outputs, dev_gold, beta=1.0)
                scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)
                for rel in RELATIONS:
                    if curr_best_scores[rel] is None or scores_per_relation[rel]['f1'] > curr_best_scores[rel]['f1']:
                        curr_best_scores[rel] = scores_per_relation[rel]
                        curr_best_thresholds[rel] = curr_t
                pass
                curr_t += step

            curr_best_scores["***Average***"] = {
                "p": sum([x["p"] for x in curr_best_scores.values()]) / len(
                    curr_best_scores),
                "r": sum([x["r"] for x in curr_best_scores.values()]) / len(
                    curr_best_scores),
                "f1": sum([x["f1"] for x in curr_best_scores.values()]) / len(
                    curr_best_scores),
            }
            print(pd.DataFrame(curr_best_scores).transpose().round(3))
            for rel in prompt_weights_learner.weights:
                print(f"rel: {rel}; thres: {curr_best_thresholds[rel]}; weights: {prompt_weights_learner.weights[rel].tolist()};")

            if best_f1 is None or curr_best_scores['***Average***']['f1'] > best_f1:
                best_f1 = curr_best_scores['***Average***']['f1']
                best_scores_per_relation = curr_best_scores
                torch.save(prompt_weights_learner.state_dict(), os.path.join(model_out_dir, 'best.ckpt'))

    print(f"Best scores: ")
    print(pd.DataFrame(best_scores_per_relation).transpose().round(3))

    return


def run_single_weights(model_names: dict, model_rel_mapping: dict, input_fn: str, top_k: int, gpu_id: int,
               use_softmax: bool, prompt_creator: PromptCreator, tagger_isalive_res_fn: str,
               tagger_isindep_res_fn: str, weight_model_path: str):
    eval_raw_outputs, eval_raw_used_prompts = run(model_names, model_rel_mapping, input_fn, top_k,
                                                    {rel: -10 for rel in RELATIONS},
                                                    {rel: None for rel in RELATIONS}, gpu_id=gpu_id,
                                                    use_softmax=use_softmax, prompt_creator=prompt_creator,
                                                    tagger_isalive_res_fn=tagger_isalive_res_fn,
                                                    tagger_isindep_res_fn=tagger_isindep_res_fn)
    gt_rows = read_lm_kbc_jsonl(input_fn)

    eval_dataset = WeightsDataset(eval_raw_outputs, gt_rows)
    num_prompts_per_rel = prompt_creator.get_num_prompts_per_rel()
    prompt_weights_learner = PromptWeights(num_prompts_per_rel)
    prompt_weights_learner.load_state_dict(torch.load(weight_model_path))
    prompt_weights_learner.eval()

    raw_outputs = []
    total_dev_loss = 0.0
    for eidx, entry in enumerate(eval_dataset):
        rel, dct, gld, subj = entry
        objs, loss = prompt_weights_learner(rel, dct, gld)
        raw_outputs.append({'subj': subj, 'rel': rel, 'objs': objs})
        if loss is not None:
            total_dev_loss += loss

    begin = 0.01
    end = 0.95
    step = 0.01

    curr_best_thresholds = {rel: None for rel in RELATIONS}
    curr_best_scores = {rel: None for rel in RELATIONS}

    curr_t = begin
    while curr_t < end:
        curr_outputs = []
        for curr_raw in raw_outputs:
            objs, scrs = filter_objs(curr_raw['objs'], curr_t, curr_raw['rel'], None)
            curr_single_o = {
                'SubjectEntity': curr_raw['subj'],
                'Relation': curr_raw['rel'],
                'ObjectEntities': objs,
                'Scores': scrs
            }
            curr_outputs.append(curr_single_o)
        scores_per_sr_pair = evaluate_per_sr_pair(curr_outputs, gt_rows, beta=1.0)
        scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)
        for rel in RELATIONS:
            if curr_best_scores[rel] is None or scores_per_relation[rel]['f1'] > curr_best_scores[rel]['f1']:
                curr_best_scores[rel] = scores_per_relation[rel]
                curr_best_thresholds[rel] = curr_t
        pass
        curr_t += step

    curr_best_scores["***Average***"] = {
        "p": sum([x["p"] for x in curr_best_scores.values()]) / len(
            curr_best_scores),
        "r": sum([x["r"] for x in curr_best_scores.values()]) / len(
            curr_best_scores),
        "f1": sum([x["f1"] for x in curr_best_scores.values()]) / len(
            curr_best_scores),
    }
    print(pd.DataFrame(curr_best_scores).transpose().round(3))

    eval_final_outputs = []
    for curr_raw in raw_outputs:
        objs, scrs = filter_objs(curr_raw['objs'], curr_best_thresholds[curr_raw['rel']], curr_raw['rel'], None)
        curr_single_o = {
            'SubjectEntity': curr_raw['subj'],
            'Relation': curr_raw['rel'],
            'ObjectEntities': objs,
            'Scores': scrs
        }
        eval_final_outputs.append(curr_single_o)

    return eval_final_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Probe a Language Model and "
                    "Run the Baseline Method on Prompt Outputs"
    )

    parser.add_argument(
        "-m",
        "--raw_model",
        type=str,
        default=None,
        help="HuggingFace model name (default: bert-large-cased)",
    )
    parser.add_argument('--tuned_ckpt', type=str, default=None)
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="trial_1.2_dependency_mined_blc",
    )
    parser.add_argument(
        "-j",
        "--job_name",
        type=str,
        default="search_thres"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="dev",
    )
    parser.add_argument(
        '--comments',
        type=str,
        default=''
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default='./data/%s.jsonl',
        help="Input test file (required)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default='./outputs/%s%s_%s.jsonl',
        help="Output file (required)",
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=100,
        help="Top k prompt outputs (default: 100)",
    )
    parser.add_argument(
        "-t",
        "--threshold_fn",
        type=str,
        default='./thresholds/thres_%s%s_F%.1f.json',
        help="Probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        type=int,
        default=-1,
        help="GPU ID, (default: -1, i.e., using CPU)"
    )
    parser.add_argument(
        '-s',
        '--use_softmax',
        type=int,
        default=1,
        help='1 for using softmax to normalize the logits, 0 for no normalization.'
    )
    parser.add_argument(
        '--exit_criterion',
        type=float,
        default=0.01,
        help='When the difference between two thresholds is within this criterion, we stop the search and stand by the '
             'better of the two.'
    )

    parser.add_argument('--isalive_res_fn', type=str, default=None)
    parser.add_argument('--isindep_res_fn', type=str, default=None)
    parser.add_argument('--relaxed_thres', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--search_sticky', action='store_true')

    parser.add_argument('--prompt_esb_mode', type=str, default='cmb')

    # arguments below are for learning weights among retrieved prompts
    parser.add_argument('--lw_num_epochs', type=int, default=10)
    parser.add_argument('--lw_lr', type=float, default=5e-3)
    parser.add_argument('--lw_update_every', type=int, default=32, help='update parameters after how many entries;')
    parser.add_argument('--lw_model_out_dir', type=str, default='./weights_dir/%d_%d_ckpts/')
    parser.add_argument('--lw_cache_usage_mode', type=str, default=None)

    args = parser.parse_args()
    assert args.use_softmax in [0, 1]
    assert args.subset in ['train', 'dev', 'test']
    assert args.prompt_esb_mode in ['cmb', 'rpc']

    args.use_softmax = True if args.use_softmax == 1 else False

    if args.relaxed_thres != 1.0:
        print(f"Relaxing the best thresholds by ratio of {args.relaxed_thres}")
        args.comments += f"_relaxed%.2f" % args.relaxed_thres
    args.output = args.output % (args.version, args.comments, args.subset)
    args.threshold_fn = args.threshold_fn % (args.version, args.comments, args.beta)
    args.lw_model_out_dir = args.lw_model_out_dir % (args.lw_num_epochs, args.lw_update_every)
    os.makedirs(args.lw_model_out_dir, exist_ok=True)

    if 'baseline' in args.version:
        prompt_creator = PromptCreator(prompt_fn=None, fixed_prompt_mode='baseline')
    elif 'dpdmined' in args.version:
        prompt_creator = PromptCreator(prompt_fn=f'./prompts/dependency_based_prompts_filtered_{args.prompt_esb_mode}.jsonl', fixed_prompt_mode=None)
    elif 'spanmined' in args.version:
        prompt_creator = PromptCreator(prompt_fn=f'./prompts/middle_word_prompts_filtered_{args.prompt_esb_mode}.jsonl', fixed_prompt_mode=None)
    elif 'cmbmined' in args.version:
        prompt_creator = PromptCreator(prompt_fn=f'./prompts/combined_mined_prompts_filtered_{args.prompt_esb_mode}.jsonl', fixed_prompt_mode=None)
    else:
        prompt_creator = PromptCreator(prompt_fn=None, fixed_prompt_mode='trial')

    # model_names = {'default': '../lms/bert-large-cased',
    #                'ChemicalCompoundElement': '/home/teddy/lmkbc_checkpoints/mlm_checkpoints_trial_1.2_cmbmined_blc_'
    #                                           'joint2_withsoftmax_whatcmb_lr1e-5_10_0_silver_C_sizeall_crels_ChemicalCompoundElement/best_ckpt',
    #                'CompanyParentOrganization': '../lms/bert-large-cased',
    #                'CountryBordersWithCountry': '../lms/bert-large-cased',
    #                'CountryOfficialLanguage': '../lms/bert-large-cased',
    #                'PersonCauseOfDeath': '../lms/bert-large-cased',
    #                'PersonEmployer': '/home/teddy/lmkbc_checkpoints/mlm_checkpoints_trial_1.2_cmbmined_blc_'
    #                                   'joint2_withsoftmax_whatcmb_lr1e-5_10_0_silver_B_sizeall_crels_PersonEmployer/best_ckpt',
    #                'PersonInstrument': '../lmkbc_checkpoints/mlm_checkpoints_lr5e-6_10_0/best_ckpt',
    #                'PersonLanguage': '../lmkbc_checkpoints/mlm_checkpoints_lr5e-6_10_1/best_ckpt',
    #                'PersonPlaceOfDeath': '../lms/bert-large-cased',
    #                'PersonProfession': '../lmkbc_checkpoints/mlm_checkpoints_lr5e-6_10_0/best_ckpt',
    #                'RiverBasinsCountry': '/home/teddy/lmkbc_checkpoints/mlm_checkpoints_trial_1.2_cmbmined_blc_'
    #                                      'joint2_withsoftmax_whatcmb_lr5e-6_10_0_silver_C_sizeall_crels_RiverBasinsCountry/best_ckpt',
    #                'StateSharesBorderState': '/home/teddy/lmkbc_checkpoints/mlm_checkpoints_trial_1.2_cmbmined_blc_'
    #                                          'joint2_withsoftmax_whatcmb_lr5e-6_10_0_silver_C_size500_crels_StateSharesBorderState/best_ckpt'
    #                }

    model_names = {'default': '../lms/bert-large-cased',
                   'ChemicalCompoundElement': '/home/teddy/lmkbc_checkpoints/mlm_checkpoints_trial_1.2_cmbmined_blc_'
                                              'joint2_withsoftmax_whatcmb_lr1e-5_10_0_silver_B_sizeall_crels_ChemicalCompoundElement/best_ckpt',
                   'CompanyParentOrganization': '../lms/bert-large-cased',
                   'CountryBordersWithCountry': '../lms/bert-large-cased',
                   'CountryOfficialLanguage': '../lms/bert-large-cased',
                   'PersonCauseOfDeath': '../lms/bert-large-cased',
                   'PersonEmployer': '/home/teddy/lmkbc_checkpoints/mlm_checkpoints_trial_1.2_cmbmined_blc_'
                                     'joint2_withsoftmax_whatcmb_lr1e-5_10_0_silver_B_sizeall_crels_PersonEmployer/best_ckpt',
                   'PersonInstrument': '../lmkbc_checkpoints/mlm_checkpoints_lr5e-6_10_0/best_ckpt',
                   'PersonLanguage': '../lmkbc_checkpoints/mlm_checkpoints_lr5e-6_10_1/best_ckpt',
                   'PersonPlaceOfDeath': '../lms/bert-large-cased',
                   'PersonProfession': '../lmkbc_checkpoints/mlm_checkpoints_lr5e-6_10_0/best_ckpt',
                   'RiverBasinsCountry': '/home/teddy/lmkbc_checkpoints/mlm_checkpoints_trial_1.2_cmbmined_blc_'
                                         'joint2_withsoftmax_whatcmb_lr5e-6_10_0_silver_B_sizeall_crels_RiverBasinsCountry/best_ckpt',
                   'StateSharesBorderState': '/home/teddy/lmkbc_checkpoints/mlm_checkpoints_trial_1.2_cmbmined_blc_'
                                             'joint2_withsoftmax_whatcmb_lr5e-6_10_0_silver_B_size500_crels_'
                                             'StateSharesBorderState_noredundanttypes/best_ckpt'
                   }


    # for rel in MODELIDS_REL_MAPPING:
    #     if args.raw_model is None:
    #         model_names[rel] = args.tuned_ckpt
    #         if 'default' not in model_names:
    #             model_names['default'] = args.tuned_ckpt
    #     else:
    #         if 'default' not in model_names:
    #             model_names['default'] = args.raw_model
    #         if MODELIDS_REL_MAPPING[rel] == 0 or args.tuned_ckpt is None:
    #             model_names[rel] = args.raw_model
    #         else:
    #             assert MODELIDS_REL_MAPPING[rel] == 1
    #             model_names[rel] = args.tuned_ckpt

    if args.job_name == 'search_thres':
        print(f"Searching thresholds on the {args.subset} subset!")
        args.input = args.input % args.subset

        best_thresholds, best_sticky_ratios, assemble_best_numprompts, best_results, best_scores_idxes = \
            search_best_thresholds(model_names, MODELIDS_REL_MAPPING, args.input, args.top_k, args.gpu_id,
                                    args.use_softmax, prompt_creator=prompt_creator, tagger_isalive_res_fn=args.isalive_res_fn,
                                    tagger_isindep_res_fn=args.isindep_res_fn, beta=args.beta, search_sticky=args.search_sticky) # args.exit_criterion)

        with open(args.threshold_fn, 'w', encoding='utf8') as tfp:
            metadata = {
                'thresholds': best_thresholds,
                'sticky_ratios': best_sticky_ratios,
                'num_prompts': assemble_best_numprompts,
                'prompt_order': best_scores_idxes
            }
            json.dump(metadata, tfp, indent=4, ensure_ascii=False)
        with open(args.output, 'w', encoding='utf8') as ofp:
            for result in best_results:
                ofp.write(json.dumps(result, ensure_ascii=False) + "\n")
    elif args.job_name == 'run_single':
        assert args.subset in ['dev', 'test']
        args.input = args.input % args.subset
        with open(args.threshold_fn, 'r', encoding='utf8') as tfp:
            item = json.load(tfp)
            curr_thresholds = item['thresholds']
            curr_sticky_ratios = item['sticky_ratios']
            curr_numprompts = item['num_prompts']
            prompt_order_idxes = item['prompt_order']
        results = run_single_ranking_avg(model_names, MODELIDS_REL_MAPPING, args.input, args.top_k,
                             curr_thresholds, curr_sticky_ratios, curr_numprompts=curr_numprompts,
                             prompt_order_idxes=prompt_order_idxes, gpu_id=args.gpu_id, use_softmax=args.use_softmax,
                             prompt_creator=prompt_creator, tagger_isalive_res_fn=args.isalive_res_fn,
                             tagger_isindep_res_fn=args.isindep_res_fn, relaxed_thres=args.relaxed_thres, beta=args.beta)
        # Save the results
        logger.info(f"Saving the results to \"{args.output}\"...")
        with open(args.output, "w", encoding='utf8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    elif args.job_name == 'find_weight':
        args.train_input_fn = args.input % 'train'
        args.dev_input_fn = args.input % 'dev'
        train_cache_fn = args.input % 'train_lw_output_cache'
        dev_cache_fn = args.input % 'dev_lw_output_cache'

        with open(args.threshold_fn, 'r', encoding='utf8') as tfp:
            item = json.load(tfp)
            prompt_order_idxes = item['prompt_order']
            prompts_per_rel = {rel: [x[2] for x in prompt_order_idxes[rel][:10]] for rel in RELATIONS}
            prompt_creator.prompt_templates = prompts_per_rel
        learn_weights(model_names, MODELIDS_REL_MAPPING, args.train_input_fn, args.dev_input_fn,
                      train_cache_fn, dev_cache_fn, cache_usage_mode=args.lw_cache_usage_mode, top_k=args.top_k, gpu_id=args.gpu_id,
                      use_softmax=args.use_softmax, prompt_creator=prompt_creator, tagger_isalive_res_fn=args.isalive_res_fn,
                      tagger_isindep_res_fn=args.isindep_res_fn, num_epochs=args.lw_num_epochs, lr=args.lw_lr,
                      grad_acc_steps=args.lw_update_every, model_out_dir=args.lw_model_out_dir)
    elif args.job_name == 'run_single_weight':
        args.eval_input_fn = args.input % args.subset
        with open(args.threshold_fn, 'r', encoding='utf8') as tfp:
            item = json.load(tfp)
            prompt_order_idxes = item['prompt_order']
            prompts_per_rel = {rel: [x[2] for x in prompt_order_idxes[rel][:10]] for rel in RELATIONS}
            prompt_creator.prompt_templates = prompts_per_rel
        run_single_weights(model_names, MODELIDS_REL_MAPPING, args.eval_input_fn, top_k=args.top_k, gpu_id=args.gpu_id,
                           use_softmax=args.use_softmax, prompt_creator=prompt_creator, tagger_isalive_res_fn=args.isalive_res_fn,
                           tagger_isindep_res_fn=args.isindep_res_fn, weight_model_path=os.path.join(args.lw_model_out_dir, 'best.ckpt'))
    else:
        raise AssertionError


if __name__ == '__main__':
    main()
