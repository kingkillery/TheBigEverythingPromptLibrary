# Voyage AI Embeddings Guide

## Description
Concise how-to for generating text embeddings using Voyage AI with Anthropic's API. Covers the Python client, HTTP requests, and practical retrieval examples.

## Source
[Anthropic Cookbook - How to create embeddings](https://github.com/anthropics/anthropic-cookbook/blob/main/third_party/VoyageAI/how_to_create_embeddings.md)

## Overview
- Install the `voyageai` Python package and set the `VOYAGE_API_KEY` environment variable.
- Create a `voyageai.Client` instance to embed documents or queries.
- Use `embed(texts, model="voyage-2", input_type="document")` for documents and `input_type="query"` for search queries.
- Embeddings are 1,024 or 1,536 dimensional vectors depending on the model.
- You can also call the HTTP API endpoint `https://api.voyageai.com/v1/embeddings` with JSON payloads for language-agnostic usage.
- Combine embeddings with vector search libraries or databases to implement Retrieval-Augmented Generation (RAG).

