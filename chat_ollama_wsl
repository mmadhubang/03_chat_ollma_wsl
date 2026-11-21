import json
import gradio as gr
import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:1.5b"


def _extract_text(content) -> str:
  """Normalize Gradio message content (string or list of content blocks) to plain text."""
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts = []
    for item in content:
      if isinstance(item, str):
        parts.append(item)
      elif isinstance(item, dict) and item.get("type") == "text":
        parts.append(item.get("text", ""))
    return "".join(parts)
  return str(content)


def chat_with_ollama(message: str, history: list):
  messages = []
  for turn in history:
    role = turn.get("role")
    text = _extract_text(turn.get("content", ""))
    if role in ("user", "assistant") and text:
      messages.append({"role": role, "content": text})

  messages.append({"role": "user", "content": message})

  payload = {
      "model": MODEL_NAME,
      "messages": messages,
      "stream": True,
  }

  partial_reply = ""
  try:
    with requests.post(
        OLLAMA_CHAT_URL, json=payload, stream=True, timeout=120
    ) as resp:
      resp.raise_for_status()
      for line in resp.iter_lines(decode_unicode=True):
        if not line:
          continue
        data = json.loads(line)
        chunk = data.get("message", {}).get("content", "")
        if chunk:
          partial_reply += chunk
          yield partial_reply
        if data.get("done"):
          break
  except requests.exceptions.ConnectionError:
    yield (
        "Could not connect to Ollama at `http://localhost:11434`. "
        "Make sure Ollama is running inside WSL and port 11434 is reachable."
    )
  except Exception as exc:
    yield f"Error talking to Ollama: {exc}"


demo = gr.ChatInterface(
    fn=chat_with_ollama,
    title="Qwen 2.5 (1.5B) — WSL Ollama + Windows Gradio",
    description="Chatting from Windows Gradio UI with Ollama running in WSL.",
)

if __name__ == "__main__":
  demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
