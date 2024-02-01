import asyncio
import json
import logging
import random
import time
from typing import Callable, Dict, List, Tuple

import tiktoken


class AIBenchRunner:
    """
    A benchmark runner for asynchronous LLM inference endpoints.

    This class facilitates benchmarking asynchronous functions that take a
    string as input and return a string asynchronously (streaming). It supports
    concurrent execution of requests and provides metrics such as
    time-to-first-token (ttft), end-to-end latency (e2e_latency), inter token
    latency (itl), cold start time, prompt tokens, output tokens, total tokens,
    and failed queries.
    """

    def __init__(self, fn: Callable, load: int, input_policy: str, seed: int = 42):
        """
        Initialize the AIBenchRunner.

        :param fn: Callable
            The function to benchmark. Takes as input a string and returns a
            string asynchronously (streaming).
        :param load: int
            The number of concurrent requests to run.
        :param input_policy: str
            The input policy to use (short or long).
        :param seed: int, optional
            The seed for the random number generator. Default is 42.
        """
        self.fn = fn
        self.load = load
        assert input_policy in {"short", "long"}
        self.input_policy = input_policy

        self.ttft = []
        self.end_to_end_latency = []
        self.cold_start = 0
        self.prompt_tokens = []
        self.output_tokens = []
        self.total_tokens = []
        self.failed_queries = 0

        self.prompt_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        random.seed(seed)

    @property
    def itl(self) -> List[float]:
        """
        Compute the inter-token latency.

        :return: List[float]
            The inter-token latency.
        """
        return [
            (e2e_lat - ttft) / (o_tks - 1)
            for e2e_lat, ttft, o_tks in zip(
                self.end_to_end_latency,
                self.ttft,
                self.output_tokens,
            )
        ]

    @property
    def output_tks_per_sec(self) -> List[float]:
        """
        Compute the output tokens per second based on ITL.

        :return: List[float]
            The output tokens per second.
        """
        return [1000 / i for i in self.itl]

    def as_dict(self) -> Dict:
        """
        Return the stored metrics as a dictionary.

        :return: Dict
            The metrics as a dictionary.
        """
        return {
            "load": self.load,
            "input_policy": self.input_policy,
            "ttft": self.ttft,
            "e2e_latency": self.end_to_end_latency,
            "itl": self.itl,
            "cold_start": self.cold_start,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "output_tks_per_sec": self.output_tks_per_sec,
            "failed_queries": self.failed_queries,
        }

    async def unpack_metrics(self):
        """Unpack the metrics from the results queue."""
        while not self.results_queue.empty():
            req_result: dict = await self.results_queue.get()
            self.ttft.append(req_result["ttft"])
            self.end_to_end_latency.append(req_result["e2e_latency"])
            self.prompt_tokens.append(req_result["prompt_tokens"])
            self.output_tokens.append(req_result["output_tokens"])
            self.total_tokens.append(req_result["total_tokens"])
            self.failed_queries += req_result["failed_queries"]
            self.results_queue.task_done()

    @staticmethod
    def _get_samples(filename: str) -> List[Tuple[str, int]]:
        """
        Retrieve samples from a JSON file.

        :param filename: str
            The name of the file to read.

        :return: List[Tuple[str, int]]
            A list of tuples containing cleaned prompts and their lengths.
        """
        with open(filename, "r") as file:
            data = json.load(file)
            samples = [
                (item["prompt"].replace("\n", ""), item["length"]) for item in data
            ]
        return samples

    def prepare_prompts(self) -> List[str]:
        """
        Prepare the prompts for the benchmark.

        :return: List[str]
            The prepared prompts.
        """
        samples_fname = f"prompts_{self.input_policy}.json"
        prompts = random.sample(self._get_samples(samples_fname), self.load)
        all_prompts = []
        for prompt in prompts:
            count = {random.randint(0, int((4096 - prompt[1]) / prompt[1]))}
            preamble = f"Repeat the following line {count} times without generating the EOS token earlier than that: \n"
            all_prompts.append(preamble + prompt[0])
        return all_prompts

    def _max_token_sampler(self) -> int:
        """
        Sample the maximum number of tokens to generate.

        :return: int
            The maximum number of tokens to generate.
        """
        if self.input_policy == "short":
            return int(random.normalvariate(200, 20))
        return int(random.normalvariate(1000, 100))

    async def compute_metrics(self) -> None:
        """Compute the metrics for an individual request."""
        prompt = await self.prompt_queue.get()
        max_tokens = self._max_token_sampler()
        completions = []
        metrics_dict = {}

        result = self.fn(prompt=prompt, max_tokens=max_tokens, stream=True)

        # Store failed queries if any
        metrics_dict["failed_queries"] = 0
        if result is None:
            metrics_dict["failed_queries"] = 1
            return

        # Streaming response loop
        start_time = time.perf_counter()
        async for part in result.generator():
            completions.append(
                {
                    "content": part["choices"][0]["delta"]["content"],
                    "reception_time": time.perf_counter(),
                },
            )
            await asyncio.sleep(0)
        end_time = time.perf_counter()

        # Compute and store metrics
        content = "".join(
            [
                completion["content"]
                for completion in completions
                if completion["content"] is not None
            ],
        )
        metrics_dict["ttft"] = (completions[0]["reception_time"] - start_time) * 1000
        metrics_dict["e2e_latency"] = (end_time - start_time) * 1000
        metrics_dict["prompt_tokens"] = len(self.tokenizer.encode(prompt))
        metrics_dict["output_tokens"] = len(self.tokenizer.encode(content))
        metrics_dict["total_tokens"] = (
            metrics_dict["prompt_tokens"] + metrics_dict["output_tokens"]
        )
        await self.results_queue.put(metrics_dict)
        self.prompt_queue.task_done()

    async def check_coldstart(self, threshold: float) -> float:
        """
        Check if the endpoint suffers from a cold start.

        :param threshold: float
            The threshold for the cold start.

        :return: float
            The cold start time.
        """
        prompt = "2+2 is "
        completions = []

        result, _ = self.fn(prompt=prompt, max_tokens=10, stream=True)

        if result is None:
            logging.warning("Run during cold start failed")
            return 0

        start_time = time.perf_counter()
        async for part in result.generator():
            completions.append(
                {
                    "content": part["choices"][0]["delta"]["content"],
                    "reception_time": time.perf_counter(),
                },
            )
            await asyncio.sleep(0)
        first_token_time = completions[0]["reception_time"]
        try:
            second_token_time = completions[1]["reception_time"]
        except IndexError:
            second_token_time = 0

        cold_start = first_token_time - start_time
        if (
            cold_start > threshold
            and (second_token_time - first_token_time) * 10 <= cold_start
        ):
            return cold_start

        return 0

    async def __call__(self) -> Dict:
        """
        Run the benchmark.

        This method performs the following steps:
        1. Checks for cold start by calling the `check_coldstart` method.
        2. Prepares prompts and adds them to the prompt queue.
        3. Initiates tasks for the concurrent requests to compute metrics.
        4. Waits for all concurrent requests to complete.
        5. Unpacks computed metrics into the corresponding attributes.
        6. Returns the computed metrics as a dictionary.

        Note:
            This method is designed to be asynchronous and should be awaited when called.

        :return: Dict
            Computed metrics as a dictionary.
        """
        self.cold_start = await self.check_coldstart(threshold=15)
        concurrent_requests = []
        for prompt in self.prepare_prompts():
            await self.prompt_queue.put(prompt)
        for _ in range(self.load):
            concurrent_requests.append(asyncio.create_task(self.compute_metrics()))
        await asyncio.gather(*concurrent_requests)
        await self.unpack_metrics()
        return self.as_dict()
