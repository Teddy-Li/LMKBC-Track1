import argparse
import json
import logging
import random

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

from file_io import read_lm_kbc_jsonl

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SUBJ_PLACEHOLDERS = {
    "PersonPlaceOfDeath": 'person',
    "PersonCauseOfDeath": 'person',
    "CompanyParentOrganization": 'company',
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
            next_layer_res[k] /= precon_key_prior[k]
    else:
        pass
    res_sum = sum(next_layer_res.values())
    next_layer_res = {k: x / res_sum for k, x in next_layer_res.items()}
    return next_layer_res


def filter_objs(output, threshold):
    banned_words = ['I', 'you', 'he', 'she', 'they', 'it', 'we', 'me', 'him', 'her', 'us', 'them', 'mine', 'yours',
                    'his', 'hers', 'ours', 'theirs', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves',
                    'yourselves', 'themselves',
                    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'many', 'some', 'much', 'most', 'any']

    curr_objs, curr_scores = [], []
    for key in output:
        if output[key] > threshold and key not in banned_words:
            curr_objs.append(key)
            curr_scores.append(output[key])
        else:
            pass
    return curr_objs, curr_scores


def create_prompt(subject_entity, relation, mask_token):
    # depending on the relation, we fix the prompt
    # prompts: [step1, step2, ...]
    # step: {"precondition1": [prompt1, prompt2, ...], "precondition2": [prompt3, prompt4], ...}
    if relation == "CountryBordersWithCountry":
        prompts = [{"": [subject_entity + " shares border with {}.".format(mask_token)]}]
    elif relation == "CountryOfficialLanguage":
        prompts = [{"": ["The official language of " + subject_entity + " is {}.".format(mask_token),
                         "{} is the official language of ".format(mask_token) + subject_entity + "."]}]
    elif relation == "StateSharesBorderState":
        prompts = [{    "": [subject_entity + ", as a place, is a {}.".format(mask_token)]
                    },
                   {    "state": [subject_entity + " state shares border with {} state.".format(mask_token)],
                        "province": [subject_entity + " province shares border with {} province.".format(mask_token)],
                        "region": [subject_entity + " region shares border with {} region.".format(mask_token)],
                        "department": [subject_entity + " department shares border with {} department.".format(mask_token)],
                        "city": ["The city of " + subject_entity + " shares border with the city of {}.".format(mask_token)]
                    }
                   ]
    elif relation == "RiverBasinsCountry":
        prompts = [{"": [subject_entity + " river is in  {}.".format(mask_token)]}]
    elif relation == "ChemicalCompoundElement":
        prompts = [{"": ["The chemical compound " + subject_entity + " consists of {}, which is an element.".format(mask_token)]}]
    elif relation == "PersonLanguage":
        prompts = [{"": [subject_entity + " speaks in {}.".format(mask_token)]}]
    elif relation == "PersonProfession":
        prompts = [{"": [subject_entity + " is a {} by profession.".format(mask_token)]}]
    elif relation == "PersonInstrument":
        prompts = [{"": ["The musician " + subject_entity + " plays {}, which is an instrument.".format(mask_token)]}]
    elif relation == "PersonEmployer":
        prompts = [{"": ["The person " + subject_entity + " is an employer at {}, which is a company.".format(mask_token)]}]
    elif relation == "PersonPlaceOfDeath":
        prompts = [{"": ["Has " + subject_entity + " died? {}.".format(mask_token)]}, {"yes": [subject_entity + " died at the place {}.".format(mask_token)], "no": None}]
    elif relation == "PersonCauseOfDeath":
        prompts = [{"": ["Has " + subject_entity + " died? {}.".format(mask_token)]}, {"yes": [subject_entity + " died due to {}.".format(mask_token)], "no": None}]
    elif relation == "CompanyParentOrganization":
        prompts = [{"": ["Does an organization own " + subject_entity + "? {}.".format(mask_token)]}, {"yes": ["The parent organization of the company " + subject_entity + " is the company {}.".format(mask_token)], "no": None}]
    else:
        raise AssertionError
    return prompts


def gather_key_priors(pipe, mask_token):
    result = {}

    for relation in SUBJ_PLACEHOLDERS:
        prompts = create_prompt(SUBJ_PLACEHOLDERS[relation], relation, mask_token)
        assert len(prompts) <= 2
        prec_prompts = prompts[0][""]
        prec_keys = list(prompts[1].keys())

        prec_all_res = {}
        prec_key_res = {}

        for p in prec_prompts:
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


def run(args):
    # Load the model
    model_type = args.model
    logger.info(f"Loading the model \"{model_type}\"...")

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForMaskedLM.from_pretrained(model_type)

    pipe = pipeline(
        task="fill-mask",
        model=model,
        tokenizer=tokenizer,
        top_k=args.top_k,
        device=args.gpu
    )

    mask_token = tokenizer.mask_token

    key_priors = gather_key_priors(pipe, mask_token)

    # Load the input file
    logger.info(f"Loading the input file \"{args.input}\"...")
    input_rows = read_lm_kbc_jsonl(args.input)
    logger.info(f"Loaded {len(input_rows):,} rows.")

    # Run the model
    logger.info(f"Running the model...")
    all_outputs = []
    all_used_prompts = []

    for eidx, entry in enumerate(input_rows):
        if eidx % 100 == 0:
            print(eidx)

        entry_ps = create_prompt(subject_entity=entry['SubjectEntity'],
                                relation=entry['Relation'],
                                mask_token=mask_token)
        subject_entity_words = entry['SubjectEntity'].split()
        if len(entry_ps) > 2:
            raise NotImplementedError

        joined_outputs = None
        final_prompts = None
        precondition = ""
        for lidx, layer in enumerate(entry_ps):
            overwrite_precondition = None
            for key in layer:
                if key in subject_entity_words:
                    assert overwrite_precondition is None
                    overwrite_precondition = key

            joined_outputs = {}
            layer_ps = layer[precondition]
            final_prompts = layer_ps

            if layer_ps is None:
                precondition = ""
                joined_outputs = {}
                continue
            else:
                for p in layer_ps:
                    probe_outputs = pipe(p)
                    for ent in probe_outputs:
                        if ent['token_str'] not in joined_outputs:
                            joined_outputs[ent['token_str']] = ent['score']
                        else:
                            joined_outputs[ent['token_str']] = max(joined_outputs[ent['token_str']], ent['score'])

            if lidx == len(entry_ps) - 1:
                continue
            else:
                next_layer_keys = list(entry_ps[lidx+1].keys())
                next_layer_res = {}
                for key in next_layer_keys:
                    if key in joined_outputs:
                        next_layer_res[key] = joined_outputs[key]
                    else:
                        next_layer_res[key] = 0

                if sum(next_layer_res.values()) == 0:
                    precondition = random.choice(next_layer_keys)
                else:
                    next_layer_res = normalize_to_1(next_layer_res, key_priors[entry['Relation']] if entry['Relation'] in key_priors else None)
                    precondition = sorted(next_layer_res.items(), key=lambda x: x[1], reverse=True)[0][0]  # the key of the first entry in the reverse-sorted dict by value.
                continue
        all_outputs.append(joined_outputs)
        all_used_prompts.append(final_prompts)

    results = []
    for row, prompt, output in zip(input_rows, all_used_prompts, all_outputs):
        curr_objs, curr_scores = filter_objs(output, args.threshold)
        result = {
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "Prompt": prompt,
            "ObjectEntities": curr_objs,
            "Scores": curr_scores
        }
        results.append(result)

    # Save the results
    logger.info(f"Saving the results to \"{args.output}\"...")
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Probe a Language Model and "
                    "Run the Baseline Method on Prompt Outputs"
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="bert-large-cased",
        help="HuggingFace model name (default: bert-large-cased)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input test file (required)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
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
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=-1,
        help="GPU ID, (default: -1, i.e., using CPU)"
    )

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
