import os
import re
import argparse
from openai import OpenAI

BASE_URLS = {
    "beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "singapore": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
}

SYSTEM_PROMPT = (
    "You are a strict judge for an ongoing math-problem solving attempt.\n"
    "You will receive TASK and HISTORY (concatenated text).\n"
    "Output ONLY one integer score from 0 to 10. No extra words.\n"
    "Score reflects progress + quality.\n"
    "Anchors: 0 empty/irrelevant; 3-4 understanding/plan; 5-6 meaningful partial; "
    "7-8 near complete; 9 almost done; 10 complete and well-checked."
)

def parse_score(text: str) -> int:
    # 只要 0-10 的整数；模型偶尔可能输出换行/句号，所以用正则兜底
    m = re.search(r"\b(10|[0-9])\b", text.strip())
    if not m:
        raise ValueError(f"Cannot parse score from model output: {text!r}")
    return int(m.group(1))

def judge_score(task: str, history: str, model: str, region: str) -> int:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set (your Bailian/DashScope API key).")

    base_url = BASE_URLS.get(region.lower())
    if not base_url:
        raise RuntimeError(f"Unknown region {region}. Choose from: {list(BASE_URLS.keys())}")

    # 用 OpenAI SDK 走百炼的 OpenAI 兼容接口 :contentReference[oaicite:4]{index=4}
    client = OpenAI(api_key=api_key, base_url=base_url)

    user_input = f"TASK:\n{task.strip()}\n\nHISTORY:\n{history.strip() if history.strip() else '(empty)'}\n"

    resp = client.chat.completions.create(
        model=model,  # 例如 qwen-max / qwen-plus / qwen-turbo :contentReference[oaicite:5]{index=5}
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
        max_tokens=10,
    )

    out = (resp.choices[0].message.content or "").strip()
    return parse_score(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--history", default="")
    ap.add_argument("--history_file", default="")
    ap.add_argument("--model", default="qwen-max")
    ap.add_argument("--region", default="beijing")  # 你截图是北京，就用默认
    args = ap.parse_args()

    history = args.history
    if args.history_file:
        with open(args.history_file, "r", encoding="utf-8") as f:
            history = f.read()

    score = judge_score(args.task, history, args.model, args.region)
    print(score)  # 只输出分数

if __name__ == "__main__":
    main()
