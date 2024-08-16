"""
An extended version of the Penzai temperature sampling loop that supports stop_tokens.
The original version is available at https://github.com/google-deepmind/penzai/blob/main/penzai/models/transformer/simple_decoding_loop.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from penzai import pz
from penzai.models.transformer import sampling_mode


def temperature_sample_pyloop(
    model: sampling_mode.KVCachingTransformerLM,
    prompt: pz.nx.NamedArray,
    rng: jax.Array,
    temperature: float = 1.0,
    max_sampling_steps: int | None = None,
    stop_tokens: list[int] = [],
) -> pz.nx.NamedArray:
  """Runs temperature sampling in a Python for loop.

  Args:
    model: The converted model we are running inference with.
    prompt: A named array of prompt tokens. Must have a "seq" axis along which
      the tokens for each batch element are arranged. Should always have at
      least one non-padding token along the "seq" axis (usually the beginning-
      of-sequence token).
    rng: JAX PRNGKey to use for sampling.
    temperature: Temperature to sample at.
    max_sampling_steps: Maximum number of sampling steps to run. If None,
      samples until filling up the key-value cache.
    stop_tokens: A list of stop tokens to stop sampling at.

  Returns:
    A named array of continuations of the prompt.
  """
  if max_sampling_steps is None:
    max_sampling_steps = model.cache_len

  all_log_probs = model(prompt)

  # Find the last non-padding token; this determines where we can find the
  # next log probs.
  non_padding_mask = prompt != model.pad_id
  max_non_padding_index = pz.nx.nmap(
      lambda x: jnp.argmax(jnp.where(x, jnp.arange(x.shape[0]), -1))
  )(non_padding_mask.untag("seq"))
  next_log_probs = all_log_probs[{"seq": max_non_padding_index}]

  sampled_count = 0
  initial_cache_index = model.cache_end_index.value

  # start a counter for stop tokens in each seq in the batch
  count_stop = pz.nx.zeros({'batch': prompt.named_shape['batch']}, dtype=jnp.bool_)

  while True:
    rng, key = jax.random.split(rng)
    # Split a key across named axes:
    batched_keys = pz.nx.random_split(key, model.batch_axes)
    next_token = pz.nx.nmap(jax.random.categorical)(
        batched_keys, next_log_probs.untag("vocabulary") / temperature
    )
    sampled_count += 1
    
    # update stop counter mask
    count_stop = count_stop | pz.nx.nmap(lambda x: jnp.isin(x, jnp.array(stop_tokens, dtype=jnp.int32)))(next_token)

    # Are we done?
    if (
        model.cache_end_index.value >= model.cache_len
        or sampled_count >= max_sampling_steps
        # stop early if we have a stop token in every sequence in the batch
        or count_stop.untag('batch').all()
    ):
      break
    
    next_log_probs = model(next_token[{"seq": None}]).untag("seq").squeeze(0)
  # Add the last token we sampled (which doesn't need to be run through the
  # model).
  start = initial_cache_index
  end = model.cache_end_index.value
  final_written = pz.nx.concatenate(
      [
          model.previous_tokens.value[{"seq": pz.slice[start:end]}],
          next_token[{"seq": None}],
      ],
      "seq",
  )
  return final_written