'''
A script to save activations and text completions from a model for each prompt in a prompts json lines file.
Run with: `python scripts/save-activations.py --config configs/summarize_email-multi-gemma_2b_it.yaml --print`
Check notebooks/01-penzai-and-activation-saving.ipynb for details
'''

from __future__ import annotations
from typing import Any, List, Dict

import os
import gc
import argparse
import jax
import numpy as np

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from nanoid import generate
from tqdm import tqdm

from utils.load_config import load_config
from utils.load_model import load_model_from_checkpoint
from utils.model_sampler import ModelSampler

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_log_compiles", True)

def save_df(df, save_dir, filename):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path=f"{save_dir}/{filename}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Load configuration from YAML and run model on prompts")
    parser.add_argument('--config', type=str, help='Path to the config file.')
    parser.add_argument('--print', action='store_true', help='Print the completions to the console.')
    args = parser.parse_args()
    
    config = load_config(args.config)
    model, vocab = load_model_from_checkpoint(config)
    out_dir = config['out_dir']
    out_filename = config['out_filename']
    model_name = config['model_name']
    batch_size = config['batch_size']
    cache_len = config['cache_len']
    stop_tokens = config['stop_tokens']
    prompt_template = config['prompt_template']
    sampler = ModelSampler(model, vocab, batch_size, cache_len=cache_len, stop_tokens=stop_tokens)

    prompts_df = pd.read_json(path_or_buf=config['prompts_file'], lines=True)
    prompts = prompts_df.to_dict('records')

    df = pd.DataFrame(columns=['id', 'parent_id', 'prompt', 'prompt_type', 'completion', 'eval_completion_success_with', 'prompt_metadata', 'has_prompt_injection', 'eval_injection_success_with', 'poison_type', 'poison_metadata', 'model', 'layer_activations_metadata', 'layer_activations'])

    # ensure len is a multiple of batch_size
    len_mod = len(prompts) % batch_size
    if len_mod != 0:
        print(f"WARNING: len(prompts) is not a multiple of batch_size, truncating to {len(prompts) - len_mod} prompts")
        prompts = prompts[:-len_mod]
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i + batch_size]
        (activations, completions) = sampler.forward(map(lambda x: x["prompt"], batch), max_sampling_steps=cache_len)
        for j in range(len(completions)):
            row = batch[j]
            act, compl = activations[j], completions[j]
            prompt = prompt_template.format(prompt=row["prompt"]) if prompt_template else row["prompt"]
            new_data_df = pd.DataFrame([{
                'id': row["id"],
                'parent_id': row["parent_id"],
                'prompt_type': row["prompt_type"],

                'prompt': row["prompt"],
                'eval_completion_success_with': row["eval_completion_success_with"],
                'prompt_metadata': {**row["prompt_metadata"], **{"prompt_template": prompt_template}} if row["prompt_metadata"] else {"prompt_template": prompt_template},

                'has_prompt_injection': row["has_prompt_injection"],
                'poison_type': row["poison_type"],
                'eval_injection_success_with': row['eval_injection_success_with'],
                'poison_metadata': row['poison_metadata'],
                
                'completion': compl,
                'model': model_name,

                'layer_activations_metadata': act.named_shape,
                'layer_activations': np.array(act.unwrap('embedding', 'layer').flatten())
            }])
            if args.print:
                print(f"PROMPT: {prompt}")
                print(f"COMPLETION: {compl}")
            df = pd.concat([df, new_data_df], ignore_index=True)
            print(f"concated {len(df)} rows")
            del act, compl, new_data_df
            gc.collect()
        del activations, completions
        gc.collect()
        
        filename = f"{out_filename}.parquet"
        print(f"Saving {len(df)} rows to {out_dir}/{filename}")
        save_df(df, out_dir, filename)

if __name__ == '__main__':
    main()