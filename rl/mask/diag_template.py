"""Diagnose the system-message template fragment extraction failure."""
import sys
from transformers import AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

print(f"Loading tokenizer from: {MODEL}")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

_USER_ONLY_BASE = [{"role": "user", "content": "I am a user."}]
sys_msg = {"role": "system", "content": "You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks."}

base_str = tok.apply_chat_template(_USER_ONLY_BASE, add_generation_prompt=False, tokenize=False)
with_msg = tok.apply_chat_template([sys_msg] + _USER_ONLY_BASE, add_generation_prompt=False, tokenize=False)

print("\n===== base_str (render of [user_base]) =====")
print(repr(base_str))
print("\n===== with_msg (render of [sys_msg, user_base]) =====")
print(repr(with_msg))
print("\n===== with_msg.endswith(base_str) ? =====")
print(with_msg.endswith(base_str))

if not with_msg.endswith(base_str):
    i = 0
    while i < len(base_str) and i < len(with_msg) and with_msg[-(i+1)] == base_str[-(i+1)]:
        i += 1
    print(f"\nSuffix match length from end: {i}")
    print(f"with_msg tail (last 150): {with_msg[-150:]!r}")
    print(f"base_str tail (last 150): {base_str[-150:]!r}")

print("\n===== with enable_thinking=False =====")
try:
    b2 = tok.apply_chat_template(_USER_ONLY_BASE, add_generation_prompt=False, tokenize=False, enable_thinking=False)
    w2 = tok.apply_chat_template([sys_msg]+_USER_ONLY_BASE, add_generation_prompt=False, tokenize=False, enable_thinking=False)
    print(f"endswith base? {w2.endswith(b2)}")
    print(f"with_msg2: {w2!r}")
except Exception as e:
    print(f"enable_thinking=False error: {e!r}")

print("\n===== with system content as LIST =====")
sys_msg_list = {"role": "system", "content": [{"type": "text", "text": "You are OpenHands agent."}]}
try:
    w3 = tok.apply_chat_template([sys_msg_list]+_USER_ONLY_BASE, add_generation_prompt=False, tokenize=False)
    print(f"endswith base? {w3.endswith(base_str)}")
    print(f"with_msg3: {w3!r}")
except Exception as e:
    print(f"list-content error: {e!r}")
