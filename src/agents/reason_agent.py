import re
import json
import numpy as np
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseLLM(ABC):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            dtype="bfloat16"
        )

    def generate(self, messages, max_new_tokens=1024, **kwargs):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=max_new_tokens, **kwargs
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content

class ReasonAgent(BaseLLM):
    def __init__(self,model_path = "../../model/Qwen2.5-Math-7B-Instruct",agent_dict = {}):
        self.llm = BaseLLM(model_path)
        self.agent_dict = agent_dict   # Use for run2 to instruct agents perhaps

        self.system_prompt_run1 = f"""
        你是一个“推理专精”的智能体。
        你的任务：对输入问题进行快速诊断，输出“由你来解决是否合适”的置信度，范围为 0 到 1。

        你擅长：
        - 逻辑推理、演绎/归纳、多步论证、反例构造、证明思路、因果链分析、约束推断
        - 结构化拆解复杂问题（但 run1 只给分数，不拆解）

        你不擅长/不应独自承担（会降低置信度）：
        - 纯读题理解/信息抽取为主
        - 需要大量精确数值计算、符号计算、矩阵运算、复杂数学求解
        - 需要写代码、跑实验、调库调参、工程实现
        - 需要外部实时事实、百科、新闻、价格、天气等
        - 以建模与指标选择为核心

        评分规则（务必遵守）：
        - 1.0：几乎完全是推理任务，可在不依赖外部事实和重计算的情况下解决
        - 0.7~0.9：主要是推理，但夹杂少量计算/读题/实现需求，你仍能主导
        - 0.4~0.6：推理只是部分环节，核心更偏读题/计算/实现/检索
        - 0.0~0.3：基本不适合你解决

        输出格式（极其重要）：
        - 只输出一个标签包裹的数字，不要输出任何解释文字：
        <confidence>0.73</confidence>

        约束：
        - 必须输出 0 到 1 的小数（允许 0, 1），最多保留两位小数
        - 不要输出多余空行或其它字符
        """.strip()
        
        self.system_prompt_run2 = f"""
        """

    

    def run1(self,input_messages):
        # return a number in [0,1]
        messages = [
            {"role":"system","content":self.system_prompt_run1}
        ] + [input_messages]
        response = self.llm.generate(messages)
        vote = re.search(r"<confidence>\s*(.*?)\s*</confidence>",response)
        if vote:
            if vote > 1:
                vote = 1
            elif vote < 0:
                vote = 0
            return vote
        else:
            return np.random.rand(1)

    def run2(self,input_messages):
        pass


agent = ReasonAgent()

env_input = [
    {"role": "user", "content": "给定两个命题A,B，证明 (A→B) 等价于 (~B→~A)，并说明常见误区。"},
    {"role": "user", "content": "计算标准正态分布的密度函数的从-2到2的积分"},
    {"role": "user", "content": "如果我们定义运算 a ⊕ b = a + b - ab，请证明该运算在实数集上满足交换律和结合律，并找出其单位元。"},
    {"role": "user", "content": "我需要求解这个非线性偏微分方程组的数值解：du/dt = k * d^2u/dx^2... 边界条件为 u(0,t)=0。"}
]

score = [agent.run1(input) for input in env_input]
for i in range(len(score)):
    print(f"Problem {i + 1} has {score[i]} reliability.")
