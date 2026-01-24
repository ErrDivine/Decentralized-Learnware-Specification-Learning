import os
import json
import urllib.request
import urllib.error


class ComputeAgent:
    def __init__(self, model_name: str | None = None, base_url: str | None = None, timeout: int = 120):
        # 默认模型名：对应你原来的 Qwen2.5-3B-Instruct（百炼 openai-compat 模型名）
        if model_name is None:
            model_name = "qwen2.5-3b-instruct"
        self.model_name = model_name

        # 你已配置在环境变量里：DASHSCOPE_API_KEY
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("未检测到环境变量 DASHSCOPE_API_KEY，请先设置后再运行。")

        # base_url：默认北京；可用环境变量 DASHSCOPE_BASE_URL 覆盖
        self.base_url = (base_url or os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.timeout = timeout

        # !!! 下面 prompt 完全不动（按你的原样保留） !!!
        self.system_prompt = f"""
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

    def _post_chat_completions(self, messages, max_tokens: int = 1024, **kwargs) -> str:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": int(max_tokens),
        }

        # 可选采样参数：从 run() 透传
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
    agent = ComputeAgent()
    print(agent.run("Compute: 48/2 + 48"))
