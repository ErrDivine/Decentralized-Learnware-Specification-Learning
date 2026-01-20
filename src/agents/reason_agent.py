import re
import json
import numpy as np
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from base_llm import BaseLLM


class ReasonAgent(BaseLLM):
    def __init__(self,model_path = "../../model/Qwen2.5-3B-Instruct"):
        self.llm = BaseLLM(model_path)

        self.system_prompt = f"""
        You are a Reasoning Specialist for math problems.

        [Identity]
        Your job is to derive the solution step-by-step using logical reasoning. You turn a model/plan (if provided) into a coherent derivation.

        [Capabilities]
        1.Produce a clear sequence of reasoning steps with correct logic.
        2.Handle casework, proofs, transformations, and algebraic manipulation when it is not computationally heavy.
        3.Keep track of assumptions and domain constraints throughout.
        4.When calculations become tedious or error-prone, you may stop and explicitly state what exact computation is needed (e.g., “Compute this expression / solve this system”), rather than risking mistakes.
        5.Provide a final expression/answer if it is reasonably obtainable without heavy computation.

        [Boundaries]
        1.Do not spend effort on very large arithmetic, huge symbolic expansions, or complicated numeric approximations.
        2.Do not skip justification: every non-trivial step should be motivated.
        3.Output should be normal prose math reasoning.
        """

    
    def run(self,input_message):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_message}
        ]
        
        response_text = self.llm.generate(messages)
        return response_text


agent = ReasonAgent()
response = agent.run("Mr. Sanchez found out that 40% of his Grade 5  students got a final grade below B. How many of his students got a final grade of B and above if he has 60 students in Grade 5?")
print(response)

