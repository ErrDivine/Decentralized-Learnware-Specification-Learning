import os
import json
import urllib.request
import urllib.error


class ReasonAgent:
    def __init__(self, model_name: str | None = None, base_url: str | None = None, timeout: int = 120):
        # 模型不变：Qwen2.5-Math-7B-Instruct -> 百炼 openai-compat 模型名
        if model_name is None:
            model_name = "qwen2.5-math-7b-instruct"  # 官方支持列表包含该模型 :contentReference[oaicite:1]{index=1}
        self.model_name = model_name

        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("未检测到环境变量 DASHSCOPE_API_KEY，请先设置后再运行。")

        self.base_url = (base_url or os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.timeout = timeout

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

    def _post_chat_completions(self, messages, max_tokens: int = 1024, **kwargs) -> str:
        url = f"{self.base_url}/chat/completions"  # 官方 endpoint :contentReference[oaicite:3]{index=3}

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": int(max_tokens),
        }

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
    agent = ReasonAgent()
    print(agent.run("Solve: 2x + 3 = 11"))
