---
sidebar_position: 1
title: SAEs & Features
---

# Sparse Autoencoders & Features

A short primer on the interpretability machinery miLLM is built around. If you already know what a residual stream and an SAE feature are, skip to [How Steering Works](/concepts/steering).

## The residual stream

A transformer processes text as a sequence of vectors — one per token — that flow through the layers. Each layer reads this **residual stream**, computes something (attention, MLP), and *adds* its result back into the stream. By the final layer, the stream encodes everything the model "thinks" about the text so far.

The catch: those vectors are **dense and polysemantic**. A single dimension doesn't mean anything by itself; concepts are smeared across thousands of dimensions, and each dimension participates in many concepts. You can't read the stream directly.

## What an SAE does

A **Sparse Autoencoder** is a translation layer trained to decompose the residual stream into interpretable parts. It learns two maps:

- **Encoder** — projects a residual-stream vector (dimension `d_in`, e.g. 2304 for Gemma 2 2B) up into a much wider, *sparse* feature space (dimension `d_sae`, e.g. 16,384 or 65,536). For any given token, only a few dozen features are active.
- **Decoder** — projects feature activations back down, approximately reconstructing the original vector. Each feature owns one **decoder direction**: the vector in residual-stream space that the feature "writes."

Because the training objective forces sparsity, individual features tend to align with human-recognizable concepts: *references to dogs*, *legal language*, *the concept of deception*, *Python code*. Browsing what a feature responds to is exactly what [Neuronpedia](https://neuronpedia.org) is for.

## Why this enables causal experiments

Correlation is easy: "feature 12082 fires when the text is about dogs." The interesting claim is **causal**: "feature 12082 *makes* the model talk about dogs."

The decoder direction is the tool for testing that. If you add `strength × decoder_direction` into the residual stream, you are injecting the concept directly — and if the model's output shifts toward the concept, you've demonstrated causal influence. That intervention is [steering](/concepts/steering), and it's the core capability of miLLM.

The complementary observation tool is [monitoring](/concepts/monitoring): running the encoder on the live residual stream to see which features are active during real inference.

## SAE ↔ model compatibility

An SAE is only meaningful for the model (and the specific **layer**) it was trained on. miLLM checks compatibility at attach time:

| Check | Failure mode |
|-------|--------------|
| `d_in` must equal the model's `hidden_size` | Hard error — attach is rejected |
| Attach layer should match the SAE's trained layer | Warning — attaching layer-12 SAE to layer 5 produces noise, not features |
| Model family should match `trained_on` | Warning — a `gemma-2-2b` SAE on `gemma-2-2b-it` degrades noticeably; on Llama it's meaningless |

:::tip Where to get SAEs
The [GemmaScope](https://huggingface.co/google/gemma-scope-2b-pt-res) release covers every layer of Gemma 2 2B/9B/27B at several widths and sparsities in SAELens format, which miLLM loads natively. Community SAEs for other models are on HuggingFace — anything in SAELens format (`params.npz` + config) works.
:::

## Vocabulary

| Term | Meaning |
|------|---------|
| **Feature** | One learned direction in the SAE's sparse space; indexed `0 … d_sae−1` |
| **Activation** | How strongly a feature fires for a given token (≥ 0, post-ReLU) |
| **Decoder direction** | The residual-stream vector a feature writes; the steering handle |
| **`d_in`** | Residual-stream width of the host model |
| **`d_sae`** | Number of features in the SAE (its "width": 16k, 65k, 131k…) |
| **L0** | Average number of active features per token — the sparsity of the SAE |
| **Hook layer** | The transformer layer whose output the SAE reads/modifies |
