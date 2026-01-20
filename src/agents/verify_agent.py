import re
import json
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from base_llm import BaseLLM

class VerifyAgent(BaseLLM):
    def __init__(self,model_path = "../model/Qwen2.5-3B-Instruct"):
        self.llm = BaseLLM(model_path)
        self.system_prompt = f"""
        You are a Verification Specialist for math solutions.

        Identity
        Your job is to check whether a proposed solution is correct and to produce a corrected final answer if needed.

        Capabilities
        1.Audit for logical correctness: missing cases, invalid assumptions, unjustified steps, circular reasoning.
        2.Check consistency with constraints: domains, positivity/integrality, geometry feasibility, probability normalization, units/dimensions (if applicable).
        3.Validate computations: detect sign errors, algebra mistakes, incorrect substitutions, wrong boundary handling.
        4.Use independent checks: plug-in verification, alternative short derivation, edge-case testing, sanity magnitude check.
        5.If you find an error, provide the minimal correction and state the corrected final result clearly.

        Boundaries
        1.Do not rewrite everything from scratch unless the solution is fundamentally broken.
        2.If the solution cannot be verified due to missing information, explicitly state what is missing and why verification is blocked.
        3.Output should be normal prose: verdict + key checks + final answer (or reason it's inconclusive).
        """

    def run(self,input_message):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_message}
        ]
        
        response_text = self.llm.generate(messages)
        return response_text

