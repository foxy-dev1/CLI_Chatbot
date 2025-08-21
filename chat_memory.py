memory = []

def add_to_memory(user_msg,ai_msg):
    memory.append({"role":"user","content":user_msg})
    memory.append({"role":"assistant","content":ai_msg})
    


def get_context(no_msg_pair=2):
    if len(memory)>=no_msg_pair:
        return memory[-no_msg_pair:]
    
    else:
        return memory

