from __future__ import annotations
from typing import Any, List, Dict

import os
import gc
import jax
import orbax.checkpoint
from jax.experimental import mesh_utils

import kagglehub
import sentencepiece as spm
import treescope
import penzai
from penzai.models import transformer


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

    del flat_params, restore_args, metadata, sharding, sharding_devices, checkpointer
    gc.collect()

    print(f"Loaded model: {model_name}")
    return model, vocab