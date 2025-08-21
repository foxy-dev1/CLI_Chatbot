from model_loader import load_model
from chat_memory import add_to_memory,get_context

def main():
    try:
        print("loading model")
        tokenizer,model = load_model()
        print("model loaded")

        while True:
            user_input = input("You: ")

            if user_input.strip().lower() == "/exit":
                print("exiting Byee !!")
                break


            chat_memory = get_context(no_msg_pair=2)

            if user_input:
                chat_memory.append({"role": "user", "content": f"{user_input}"})

            inputs = tokenizer.apply_chat_template(
                chat_memory,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            outputs = model.generate(**inputs, max_new_tokens=200,pad_token_id=tokenizer.eos_token_id)
            assistant_msg = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:],skip_special_tokens=True)
            print(f"Assistant: {assistant_msg}")

            add_to_memory(user_input,assistant_msg)



    except Exception as e:
        print(f"error generating response error-> {e}")



if __name__ == "__main__":
    main()