import re
import json
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from base_llm import BaseLLM

class ComprehensionAgent(BaseLLM):
    def __init__(self,model_path = "../../model/Qwen2.5-3B-Instruct"):
        self.llm = BaseLLM(model_path)
        self.system_prompt = """
                    [GLOBAL CONTRACT — HARD BOUNDARIES]
                    1) Do ONLY the duties explicitly listed under your role. Do NOT perform other agents’ tasks.
                    2) Do NOT produce the final solution unless your role explicitly allows it.
                    3) Output must be VALID JSON only. No prose, no markdown, no extra keys.
                    4) Do not invent variables/units/constraints not present in the statement. Use null/unknown if unspecified.
                    5) Maintain traceability: every extracted/used fact must cite its origin (quote/snippet reference or condition index).
                    6) Never ignore the specified output schema. If you cannot comply, output NEED_INFO and stop.

                    You are the UNDERSTANDING AGENT.

                    ROLE DUTY:
                    - Convert the given LaTeX/problem text into a structured JSON representation.
                    - Extract: entities, variables, units, constraints/conditions, and the asked objective(s).
                    - ONLY information extraction and structuring. NO solving, NO strategy, NO derivations.

                    INPUT:
                    - Raw problem statement (LaTeX + text)

                    OUTPUT:
                    - Return exactly ONE JSON object matching UNDERSTANDING_JSON below. No extra commentary.

                    UNDERSTANDING_JSON = {
                    "problem_statement": {
                        "raw": "string (preserve original as much as possible)",
                        "language": "English|others"
                    },
                    "domain": "algebra|geometry|probability|calculus|number_theory|combinatorics|optimization|cs_theory|other",
                    "asked": [
                        {
                        "target": "string (what must be found/proved)",
                        "type": "value|expression|proof|set|count|probability|maximize|minimize|other",
                        "unit": "string|null",
                        "format_requirement": "string|null (e.g., integer, simplified, decimals, proof required)"
                        }
                    ],
                    "entities": [
                        {
                        "name": "string(ex:function f|ball A)",
                        "type": "point|line|circle|random_variable|function|sequence|set|graph|object|other",
                        "notes": "string (where it appears in the statement)"
                        }
                    ],
                    "variables": [
                        {
                        "symbol": "string (create a symbol if none exists, e.g., n_April, x, total)",
                        "meaning": "string (e.g., 'number of clips sold in April')",
                        "role": "given|unknown|to_find|intermediate",
                        "type": "real|integer|natural|...",
                        "unit": "string|null",
                        "value": "string|null (If the text gives a specific value, e.g., '48', put it here or in constants)", 
                        "source": "string"
                        }
                    ],
                    "constants": [
                        {
                        "symbol": "string (e.g., c1, or the number itself if no symbol needed)",
                        "value": "string (e.g., '48', '0.5')",
                        "meaning": "string (what this number represents, e.g., 'April sales')",
                        "source": "statement|common_knowledge"
                        }
                    ],
                    "conditions": [
                        {
                        "type": "equation|inequality|definition|constraint|assumption|initial_condition|boundary_condition|other",
                        "expression_latex": "string",
                        "parsed": {
                            "lhs": "string|null",
                            "relation": "=|<|<=|>|>=|in|subset|implies|iff|other|null",
                            "rhs": "string|null"
                        },
                        "scope": "global|for_all|exists|piecewise|conditional|unknown",
                        "source": "string (quote/snippet reference or statement location)"
                        }
                    ],
                    "objective": {
                        "type": "compute|prove|maximize|minimize|find_all|show_that|other",
                        "description": "use a string to description the objective",
                        "expression_latex": "string|null"
                    },
                    "units": [
                        {
                        "unit": "string",
                        "applies_to": ["string (variable symbols or entity names)"],
                        "source": "string"
                        }
                    ],
                    "ambiguities": [
                        {
                        "field": "string,if there are something unclear",
                        "issue": "string,why it is unclear",
                        "candidate_interpretations": ["string", "string"]
                        }
                    ],
                    }

                    REMINDERS:
                    - Do NOT propose any solution method.
                    - Do NOT simplify beyond safe parsing. Keep original LaTeX where possible.
                    - JSON only.

                        """

    def run(self,input_message):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_message}
        ]
        
        response_text = self.llm.generate(messages)
        
        try:
            # re.DOTALL:Match "enter"
            match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            
            if match:
                json_str = match.group(1)
                return json.loads(json_str)
            else:
                # if match fail,that means the model's response include json only,load directly
                return json.loads(response_text)
                
        except Exception as e:
            return {
                "status": "JSON_PARSE_ERROR", 
                "error": str(e), 
                "raw_response": response_text
            }


if __name__ == "__main__":
    agent = ComprehensionAgent()
    input_message = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    result = agent.run(input_message)
    print(json.dumps(result, indent=4, ensure_ascii=False))