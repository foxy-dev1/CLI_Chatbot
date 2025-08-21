## CLI AI Astrologer (Llama 3.2 1B Instruct)

A minimal local CLI chat app using Hugging Face Transformers with 4-bit quantization via bitsandbytes. It loads `meta-llama/Llama-3.2-1B-Instruct`, maintains a short in-memory chat history, and generates responses using the model's chat template.

**Demo**: [Watch a short CLI demo](https://drive.google.com/file/d/1gNUzcm5BxKH-xLoaYi3R8QNNU8tpBDY8/view?usp=sharing)

### Features
- **Local inference**: runs the 1B instruct model locally
- **Lightweight**: 4-bit quantization with bfloat16 compute
- **Chat memory**: short sliding window to retain recent turns
- **Simple CLI**: type to chat, `/exit` to quit

## Requirements
- **Python**: 3.9+
- **GPU (recommended)**: NVIDIA CUDA for best performance. CPU is possible but slower.
- **Dependencies**:
  - `torch`
  - `transformers`
  - `accelerate`
  - `bitsandbytes`
  - `huggingface_hub`
- **Model access**: You must accept the license for the model and (if required) authenticate.
  - Model: [Llama 3.2 1B Instruct on Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

## Installation
```bash
# 1) (Optional) create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2) Upgrade pip
python -m pip install --upgrade pip

# 3) Install dependencies
pip install -r requirements.txt

# 4) (If required) login to Hugging Face and accept the model license
huggingface-cli login
# Then visit the model page and accept the license if prompted
```

## Project structure
```text
cli/
├─ chat_memory.py       # simple in-memory chat history
├─ interface.py         # CLI entrypoint (read, generate, print)
└─ model_loader.py      # loads tokenizer & model with 4-bit quantization
```

## Usage
```bash
cd /home/leosama/ai_astrologer/cli
python interface.py
```
Example session:
```text
loading model
model loaded
You: Hello!
Assistant: Hi! How can I help you today?
You: /exit
exiting Byee !!
```

## How it works
- `model_loader.py`
  - Loads tokenizer and model from `meta-llama/Llama-3.2-1B-Instruct` with a 4-bit `BitsAndBytesConfig`, `bnb_4bit_compute_dtype=torch.bfloat16`, and `device_map="cuda"`.
- `chat_memory.py`
  - Stores turns in a simple Python list. `get_context(no_msg_pair=2)` keeps recent messages for context.
- `interface.py`
  - Prompts for user input, augments context, applies `tokenizer.apply_chat_template`, and calls `model.generate` with `max_new_tokens=200`.

## Configuration
- **Model/quantization** (`model_loader.py`):
  - Change model ID: `AutoTokenizer.from_pretrained("<model-id>")` and `AutoModelForCausalLM.from_pretrained("<model-id>")`.
  - Switch device: `device_map="cuda"` → `"auto"` or `"cpu"`.
  - Adjust quantization: update `BitsAndBytesConfig` (e.g., compute dtype or disable 4-bit).
- **Context window** (`chat_memory.py`):
  - Increase or decrease turns kept: `get_context(no_msg_pair=2)`.
- **Generation** (`interface.py`):
  - Tokens: change `max_new_tokens=200`.
  - Padding: uses `pad_token_id=tokenizer.eos_token_id`.

## Troubleshooting
- **Model download/auth errors (401/403)**:
  - Run `huggingface-cli login` and ensure you accepted the model license on its page.
- **No CUDA / GPU not found**:
  - In `model_loader.py`, set `device_map="cpu"` or `"auto"`. CPU will be slower.
- **bitsandbytes load issues**:
  - Ensure a compatible CUDA setup. On CPU-only systems, remove 4-bit quantization and load without `quantization_config` (set `device_map="cpu"`), since bitsandbytes does not support CPU-only.
- **bfloat16 errors on CPU**:
  - Use a CPU-friendly dtype or switch to GPU. For CPU, change compute dtype to `torch.float16` or disable 4-bit.
- **Out Of Memory (OOM)**:
  - Lower `max_new_tokens`, reduce context size, ensure only necessary processes are using GPU, or use a smaller model.

## Notes
- **Ephemeral memory**: chat history is kept in-process only; no persistence.
- **Exit command**: type `/exit` to quit the CLI.

## License
No license has been specified. Add a license if you intend to distribute or share this project.

