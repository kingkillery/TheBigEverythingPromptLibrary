# Promptfoo Evaluation Guide

## Description
Step-by-step instructions for running prompt evaluations using the Promptfoo framework with Anthropic models. Summarizes configuration, environment variables, and execution from the Anthropic Cookbook.

## Source
[Anthropic Cookbook - Evaluations with Promptfoo](https://github.com/anthropics/anthropic-cookbook/blob/main/skills/classification/evaluation/README.md)

## Overview
- Install Node.js and use `npx` to run Promptfoo without project setup.
- Define prompts and providers in `promptfooconfig.yaml` and reference prompts in Python functions.
- Set `ANTHROPIC_API_KEY` (and `VOYAGE_API_KEY` if needed) before running evaluations.
- Execute `npx promptfoo@latest eval -j 25` to run tests across temperature settings and datasets.
- Review the generated results to refine prompts and analyze accuracy.

