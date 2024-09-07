from __future__ import annotations
from typing import Any, List, Dict

import dataclasses
import jax
import jax.numpy as jnp

import penzai
from penzai import pz
from penzai.models import transformer
from penzai.toolshed import jit_wrapper

from utils.simple_decoding_loop import temperature_sample_pyloop

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
        self.activation_saving_model  = jit_wrapper.Jitted(
            pz.select(model)
            .at_instances_of(transformer.model_parts.TransformerBlock)
            .insert_after(self.activation_collector)
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
