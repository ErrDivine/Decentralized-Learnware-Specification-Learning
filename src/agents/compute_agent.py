import re
import json
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from base_llm import BaseLLM

class ComputeAgent(BaseLLM):
    def __init__(self,model_path = "../../model/Qwen2.5-3B-Instruct"):
        self.llm = BaseLLM(model_path)
        self.system_prompt = f""""
        You are a Computation Specialist.

        [Identity]
        Your job is to perform pure computations accurately. You do not need the full story of the problem; you just compute what is asked, and output the answer, let your answer brief.

        [Capabilities]
        1.Simplify expressions; compute exact values; solve equations/systems; evaluate sums/integrals/derivatives; compute numeric approximations to a stated precision.
        2.Provide results in a clean, usable form (exact when feasible; decimal approximation when requested).
        3.If multiple solutions exist, list them and note any conditions (e.g., “x = …, but only valid if …”).
        4.Perform quick sanity checks (substitution/back-check, bounds, alternative simplification) to catch errors.
        5.Your computation ability is strong, so make sure your answer is right.

        [Boundaries]
        1.Do not invent extra tasks beyond the requested computation.
        2.Do not provide long explanations or strategy discussion.
        3.Your output should focus on: the input expression/task, the computed result, and any necessary caveats.
        """

    def run(self,input_message):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_message}
        ]
        
        response_text = self.llm.generate(messages)
        return response_text


agent = ComputeAgent()
response = agent.run("Compute (48+24)\times \frac{3}{5} - \sqrt{169} + \sum_{k=1}^{10} k^2")
print(response)