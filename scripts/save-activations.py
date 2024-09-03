'''
A script to save activations and text completions from a model for each prompt in a prompts json lines file.
Run with: `python scripts/save-activations.py --config configs/summarize_email-multi-gemma_2b_it.yaml --print`
Check notebooks/01-penzai-and-activation-saving.ipynb for details
'''

from __future__ import annotations
from typing import Any, List, Dict

import os
import dataclasses
import gc
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from jax.experimental import mesh_utils

import kagglehub
import sentencepiece as spm
import treescope
import penzai
from penzai import pz
from penzai.models import transformer
from penzai.toolshed import token_visualization
from penzai.toolshed import jit_wrapper

import yaml
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from nanoid import generate
from tqdm import tqdm

from utils.simple_decoding_loop import temperature_sample_pyloop

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_log_compiles", True)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_from_checkpoint(config: dict):
    model_name = config['model_name']
    transformer_variant = config['transformer_variant']
    weights_dir = kagglehub.model_download(model_name)
    ckpt_path = os.path.join(weights_dir, config['ckpt_path'])

    vocab_path = os.path.join(weights_dir, 'tokenizer.model')

    vocab = spm.SentencePieceProcessor()
    vocab.Load(vocab_path)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    metadata = checkpointer.metadata(ckpt_path)

    n_devices = jax.local_device_count()
    sharding_devices = mesh_utils.create_device_mesh((n_devices,))
    sharding = jax.sharding.PositionalSharding(sharding_devices)
    restore_args = jax.tree_util.tree_map(
        lambda m: orbax.checkpoint.ArrayRestoreArgs(
            restore_type=jax.Array,
            sharding=sharding.reshape((1,) * (len(m.shape) - 1) + (n_devices,))
        ),
        metadata,
    )
    flat_params = checkpointer.restore(ckpt_path, restore_args=restore_args)

    transformer_variant_module = getattr(transformer.variants, transformer_variant)
    if transformer_variant == 'gemma':
        model = transformer_variant_module.gemma_from_pretrained_checkpoint(
            flat_params,
            upcast_activations_to_float32=True
        )
    else:
        raise ValueError(f"Unsupported transformer variant: {transformer_variant}")

    del flat_params, restore_args, metadata, sharding, sharding_devices
    gc.collect()

    print(f"Loaded model: {model_name}")
    return model, vocab

@pz.pytree_dataclass
class AppendActivationsFromLastToken(pz.nn.Layer):
    saved: pz.StateVariable[Any | None]

    def slice_before_target(self, array, target):
        # find the target
        mask = (array == target)
        indices = pz.nx.nmap(jnp.argmax)(mask.untag("seq"))
        found_mask = pz.nx.nmap(jnp.any)(mask.untag("seq"))
        
        # create a slice that's one before the target, or the last element if not found
        def get_slice(index, found):
            return jnp.where(
                jnp.logical_and(found, index >= 0),
                index - 1,
                -1
            )
        
        slices = pz.nx.nmap(get_slice)(indices, found_mask)
        
        # use the slice to index the array
        return array[{"seq": slices}]

    
    def __call__(self, value: Any, /, **_unused_side_inputs) -> Any:
        pad_id = 0
        last_token_activations = self.slice_before_target(value, pad_id)

        if self.saved.value is None:
            self.saved.value = []
        
        self.saved.value.append(last_token_activations)
                
        return value

    def finalize(self):
        if self.saved.value:
            stacked = pz.nx.stack(self.saved.value, axis_name='layer')
            # reset for next run
            self.saved.value = None
            return stacked
        return None

class ModelSampler:
    def __init__(self, model, vocab, batch_size: int = 2, cache_len: int = 100, stop_tokens: List[int] = []):
        self.vocab = vocab
        self.state = pz.StateVariable(value=None)
        self.cache_len = cache_len
        self.stop_tokens = stop_tokens

        # prepare model variations
        # a model to save activations
        self.activation_collector = AppendActivationsFromLastToken(self.state)
        patched_model  = (
            pz.select(model)
            .at_instances_of(transformer.model_parts.TransformerBlock)
            .insert_after(self.activation_collector)
        )
        self.activation_saving_model = (
            pz.select(patched_model)
            .at(lambda root: root.body)
            .apply(jit_wrapper.Jitted)
        )

        # another model to generate text completions
        inference_model = (
            transformer.sampling_mode.KVCachingTransformerLM.from_uncached(
                model, cache_len=cache_len, batch_axes={"batch": batch_size},
            )
        )
        self.model = (
            pz.select(inference_model)
            .at(lambda root: root.body)
            .apply(jit_wrapper.Jitted)
        )

    def tokenize_batch(self, prompts: List[str], include_eos: bool = True) -> pz.types.NamedArray:
        tokenized_prompts = []
        for prompt in prompts:
            tokens = [self.vocab.bos_id()] + self.vocab.EncodeAsIds(prompt)
            if include_eos:
                tokens.append(self.vocab.eos_id())
            tokenized_prompts.append(tokens)
        
        max_len = max(len(tokens) for tokens in tokenized_prompts)
        assert self.cache_len > max_len, 'prompt is too long for cache_len'
        padded_prompts = [tokens + [self.vocab.pad_id()] * (max_len - len(tokens)) for tokens in tokenized_prompts]
        
        return pz.nx.wrap(jnp.array(padded_prompts)).tag("batch", "seq")
    
    def replace_after_eos(self, seq, eos = 1):
        # find the index of the first eos in the sequence
        index = jnp.argmax(seq == eos)
        # and check if eos is actually in the sequence
        exists = jnp.any(seq == eos)
        # create a mask: 0 before the first instance, 1 after (and including)
        mask = (jnp.arange(seq.shape[0]) >= index) & exists
        # replace values after the first eos with eos, only if eos exists
        return jnp.where(mask, eos, seq)

    def detokenize(self, preds):
        # strip values after eos
        clean = pz.nx.nmap(lambda x: self.replace_after_eos(x, self.vocab.eos_id()))(preds.untag('seq')).tag('seq')
        completions = self.vocab.decode(clean.unwrap('batch', 'seq').tolist())
        
        return completions

    def forward(self, prompts: List[str], max_sampling_steps):
        # tokenize
        tokenized_prompts = self.tokenize_batch(prompts) # ('batch', 'seq')
        
        # take a single step on the model to
        # save activations at last token 
        # before predicting any new ones
        # jax.profiler.start_trace(".")
        self.activation_saving_model(tokenized_prompts)
        activations = self.activation_collector.finalize() # ('batch', 'embedding', 'layer')
        # jax.profiler.stop_trace()

        # move off the gpu and split to arrays for easier saving
        activations = pz.nx.nmap(lambda x: jax.device_put(x, jax.devices("cpu")[0]))(activations)
        activations = pz.nx.unstack(activations, "batch") # (batch, ('embedding', 'layer'))
        
        # predict new tokens
        preds = temperature_sample_pyloop(
            self.model,
            prompt=tokenized_prompts,
            rng=jax.random.key(22),
            max_sampling_steps=max_sampling_steps if max_sampling_steps else self.cache_len,
            stop_tokens=self.stop_tokens
        ) # ('batch', 'seq')

        # reset loop
        self.model.cache_end_index.value = jnp.array(0)
        
        # detokenize
        completions = self.detokenize(preds) # (batch,)
        
        return (activations, completions)

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