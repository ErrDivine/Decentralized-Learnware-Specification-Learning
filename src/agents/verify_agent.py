import os
import json
import urllib.request
import urllib.error


class VerifyAgent:
    def __init__(self, model_name: str | None = None, base_url: str | None = None, timeout: int = 120):
        # 默认模型名：对应你原来的 Qwen2.5-3B-Instruct（百炼 openai-compat 模型名）
        if model_name is None:
            model_name = "qwen2.5-3b-instruct"
        self.model_name = model_name

        # 环境变量：DASHSCOPE_API_KEY
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("未检测到环境变量 DASHSCOPE_API_KEY，请先设置后再运行。")

        # base_url：默认北京；可用环境变量 DASHSCOPE_BASE_URL 覆盖
        self.base_url = (base_url or os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.timeout = timeout

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

    def _post_chat_completions(self, messages, max_tokens: int = 1024, **kwargs) -> str:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": int(max_tokens),
        }

        # 可选：采样参数从 run() 透传
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
    agent = VerifyAgent()
    print(agent.run("Verify: If x=4 then 2x+3=11, so x=4."))
