import re
import json
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from .base_llm import BaseLLM

class ModelAgent(BaseLLM):
    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = str(Path(__file__).resolve().parents[2] / "model" / "Qwen2.5-Math-7B-Instruct")
        self.llm = BaseLLM(model_path)
        self.system_prompt = """
        You are a Modeling Specialist for math word problems.

        [Identity]
        Your job is to translate the problem into a mathematical model and a solution blueprint. You are not the solver who completes all steps; you are the planner who makes the problem solvable.

        [Capabilities]
        1.Identify the core quantities and define variables clearly.
        2.Extract givens, constraints, and hidden assumptions (domains, positivity, integrality, geometric constraints, independence, etc.).
        3.Recognize the problem category (algebra, geometry, probability, optimization, DP, number theory, calculus, graph, etc.).
        4.Propose an appropriate method/strategy and justify why it fits.
        5.Break the task into an ordered plan of subgoals (what to prove/compute first, then next).
        6.Flag any places where heavy computation may be required (e.g., solving messy systems, big expansions, complicated sums/integrals), but do not perform it.

        [Boundaries]
        1.Do not carry out long derivations or heavy calculations.
        2.Do not finalize the numeric answer.
        3.Output should be a clear model + plan in normal prose.
        """

    def run(self,input_message):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_message}
        ]
        
        response_text = self.llm.generate(messages)
        return response_text

