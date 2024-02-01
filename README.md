# AIBench LLM Endpoints

## Overview
This code provides a benchmarking runner, `AIBench-LLM`, for evaluating the performance of a large language model (LLM) inference endpoint. The benchmark measures various metrics such as Time to First Token (TTFT), End to End Latency, Inter-Token Latency (ITL), Output Tokens per Second, and more.

The AIBench Runner is in charge of collecting metrics from LLM inference endpoints for the [Unify Hub](https://unify.ai/hub). More information about the full methodology is available [here 📑](https://unify.ai/docs/hub/concepts/benchmarks.html)

Contributions and discussions around the methodology and the runner are definitely welcome, you can join the [Unify Discord](https://discord.com/invite/sXyFF8tDtm) if this sounds interesting!

## Metrics
The benchmark runner collects the following metrics:

- `load`: Number of concurrent requests.
- `input_policy`: Input policy used (short or long).
- `ttft`: Time-to-first-token for each request.
- `e2e_latency`: End-to-end latency for each request.
- `itl`: Inter-token Latency.
- `cold_start`: Cold start time (if applicable).
- `prompt_tokens`: Number of tokens in the input prompt.
- `output_tokens`: Number of tokens in the LLM output.
- `total_tokens`: Total number of tokens (input + output).
- `output_tks_per_sec`: Output tokens per second.
- `failed_queries`: Number of failed queries.

## Usage and Examples
To be added this week!
