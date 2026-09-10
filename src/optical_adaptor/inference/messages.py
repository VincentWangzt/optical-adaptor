from __future__ import annotations


def text_message(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def chat_ids(tokenizer, messages: list[dict]) -> list[int]:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer.encode(prompt, add_special_tokens=False)
