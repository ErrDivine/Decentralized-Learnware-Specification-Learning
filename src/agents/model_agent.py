import os
import json
import urllib.request
import urllib.error


class ModelAgent:
    def __init__(self, model_name: str | None = None, base_url: str | None = None, timeout: int = 120):
        # 模型不变：Qwen2.5-Math-7B-Instruct -> 百炼 openai-compat 模型名
        if model_name is None:
            model_name = "qwen2.5-math-7b-instruct"
        self.model_name = model_name

        # 你已设置在环境变量里：DASHSCOPE_API_KEY
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("未检测到环境变量 DASHSCOPE_API_KEY，请先设置后再运行。")

        self.base_url = (base_url or os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.timeout = timeout

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

    def _post_chat_completions(self, messages, max_tokens: int = 1024, **kwargs) -> str:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": int(max_tokens),
        }

        # 可选参数透传
        for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "stop", "seed"):
            if k in kwargs and kwargs[k] is not None:
                payload[k] = kwargs[k]

        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"Request failed: {e}") from e

        data = json.loads(raw)
        return data["choices"][0]["message"]["content"]

    def run(self, input_message: str, max_new_tokens: int = 1024, **kwargs) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_message},
        ]
        return self._post_chat_completions(messages, max_tokens=max_new_tokens, **kwargs)


if __name__ == "__main__":
    agent = ModelAgent()
    test = "Manny had 3 birthday cookie pies to share with his 24 classmates and his teacher, Mr. Keith.   If each of the cookie pies were cut into 10 slices and Manny, his classmates, and Mr. Keith all had 1 piece, how many slices are left?"
    print(agent.run(test))
