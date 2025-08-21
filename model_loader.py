from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


def load_model():
    """
        Returns tokenizer and model instance
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",device_map="cuda",
                                             quantization_config=bnb_config)

        return tokenizer,model
    except Exception as e:
        print(f"error loading model error-> {e}")
        return

