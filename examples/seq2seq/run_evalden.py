import argparse
import datetime
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

import transformers
from transformers import MBartTokenizer, MBartForConditionalGeneration
from utils import calculate_bleu, calculate_rouge, chunks, parse_numeric_n_bool_cl_kwargs, use_task_specific_params



DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


##--------------------------------------------------------------

def generate_summaries_or_translations(
    examples: List[str],
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    prefix=None,
    **generate_kwargs,
) -> Dict:
    """Save model.generate results to <out_file>, and return how long it took."""
    fout = Path(out_file).open("w", encoding="utf-8")
    model_name = str(model_name)
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model =  MBartForConditionalGeneration.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()

    tokenizer = MBartTokenizer.from_pretrained(model_name)
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #logger.info(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.

    start_time = time.time()
    # update config with task specific params
    use_task_specific_params(model, task)
    if prefix is None:
        prefix = prefix or getattr(model.config, "prefix", "") or ""
    for examples_chunk in tqdm(list(chunks(examples, batch_size))):
        examples_chunk = [prefix + text for text in examples_chunk]
        batch = tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest").to(device)
        summaries = model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            #**generate_kwargs,
        )
        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()
    fout.close()
    runtime = int(time.time() - start_time)  # seconds
    n_obs = len(examples)
    return dict(n_obs=n_obs, runtime=runtime, seconds_per_sample=round(runtime / n_obs, 4))


def datetime_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

##--------------------------------------------------------------



    # Takes input text, generates output, and then using reference calculates the BLEU scores.

    # The results are saved to a file and returned to the caller, and printed out unless ``verbose=False`` is passed.

    # Args:
    #     verbose (:obj:`bool`, `optional`, defaults to :obj:`True`): print results to stdout

    # Returns:
    #     a tuple: ``(scores, params}``
    #     - ``scores``: a dict of scores data ``{'bleu': 39.6501, 'n_obs': 2000, 'runtime': 186, 'seconds_per_sample': 0.093}``
    #     - ``params``: a dict of custom params, e.g. ``{'num_beams': 5, 'length_penalty': 0.8}``
   


# !python run_eval.py /content/yusuf_eymen_model \
# /content/cnn_dm_tr/test.source \
# dbart_cnn_12_6_test_gens.txt \
# --reference_path cnn_dm_tr/test.target \
# --score_path dbart_cnn_12_6_test_rouge.json \
# --n_obs 100 \
# --task summarization --bs 8 --fp16


model_name="/content/drive/MyDrive/models/my_models"
input_path="/content/transformers/examples/dummy.source"
save_path="dbart_cnn_12_6_test_gens1.txt"
reference_path="/content/transformers/examples/xsum_tr/val.target"
score_path="metrics1.json"
device=DEFAULT_DEVICE

prefix=None
task="summarization"
bs=8
n_obs=10
fp16=False


# Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate  
# args, rest = parser.parse_known_args()
# parsed_args = parse_numeric_n_bool_cl_kwargs(rest)
# if parsed_args and verbose:
#   print(f"parsed the following generate kwargs: {parsed_args}")


examples = [" " + x.rstrip() if "t5" in model_name else x.rstrip() for x in open(input_path).readlines()]
if n_obs > 0:
  examples = examples[: n_obs]
  Path(save_path).parent.mkdir(exist_ok=True)
if reference_path is None and Path(score_path).exists():
  warnings.warn(f"score_path {score_path} will be overwritten unless you type ctrl-c.")
  
runtime_metrics = generate_summaries_or_translations(
        examples,
        save_path,
        model_name,
        batch_size=bs,
        device=device,
        fp16=fp16,
        task=task,
        prefix=prefix,
        #**parsed_args,
    )
  


    # Compute scores
score_fn = calculate_bleu if "translation" in task else calculate_rouge
output_lns = [x.rstrip() for x in open(save_path).readlines()]
reference_lns = [x.rstrip() for x in open(reference_path).readlines()][: len(output_lns)]
scores: dict = score_fn(output_lns, reference_lns)
#scores.update(runtime_metrics)


print(scores)
json.dump(scores, open(score_path, "w"))

  
