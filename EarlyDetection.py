import os
import json
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from threading import Lock
import atexit

# Load API Key
load_dotenv("chatkey.env")
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# AI Instructions 
main_prompt = """
You are a formal, supportive AI medical assistant designed to assist with the EARLY DETECTION of potential neurodegenerative diseases through structured conversation.

Only continue if signs of cognitive decline (e.g., memory lapses, mood changes, language issues, tremors) are present. If the user's concerns appear unrelated (e.g., injury), politely clarify the system's purpose and conclude the conversation.

Your responsibilities are as follows:
- Begin by encouraging the user to describe any symptoms or recent changes they have experienced, especially related to memory, language, motor function, or mood.
- Ask whether there is any family history of neurodegenerative diseases to provide contextual data.
- Ask about emotional and behavioural changes, including mood swings, frustration, or a loss of motivation or interest in daily activities.
- Ask about motor-related symptoms such as clumsiness, hand tremors, slower movement, or difficulty with fine motor tasks like handwriting.
- Encourage the user to describe a recent event or daily routine in their own words. This helps you analyse natural language for signs of cognitive decline.
- Only ask one focused question at a time. Keep responses concise (ideally two sentences), and always include a targeted follow-up to collect further relevant data.
- Avoid overwhelming the user. Do not give long explanations or multiple questions in a single response.
- Actively observe the user's language for vague phrasing, reduced vocabulary, word-finding pauses, broken sentence structure, or signs of disorganised thought for indications of cognitive decline.

Throughout the conversation:
- Accumulate information over time. Combine evidence from multiple domains (language, motor, behavioural, memory) to identify meaningful patterns.
- Once a consistent pattern of symptoms is observed, you MUST determine the most likely neurodegenerative disease.
- Do NOT vaguely say "an issue" or "a condition." You MUST name a specific disease (e.g., Parkinson's disease).
- Use your understanding of symptom patterns to reason your conclusion, and then clearly state it.
"""

#Instructions purely for training mode
training_mode = """

TRAINING MODE
If you decide to suggest a likely neurodegenerative disease based on the user's symptoms, you must end your message with the exact phrase: [TRAINING COMPLETE]. This signals that feedback will follow.
"""

# Previous convo file - Chat History
mem_file = "chatmem.json"

mem_lock = Lock()

convo = []

# Track feedback state
wait_feedback = False

# LlamaIndex
Settings.embed_model = OpenAIEmbedding()

# Parse previous conversations for the memory system
def parse_mem(chatmem_path):
    if not os.path.exists(chatmem_path):
        return []
    try:
        with open(chatmem_path, "r") as file:
            sessions = json.load(file)
        docs = []
        for i, session in enumerate(sessions):
            text = f"SESSION {i+1}:\n" + "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in session])
            docs.append(Document(text=text))
        return docs
    except Exception as e:
        print(f"Error loading chat memory: {e}")
        return []

# Build search index from previous conversations
def load_index():
    docs = parse_mem(mem_file)
    if docs:
        index = VectorStoreIndex.from_documents(docs)
        search = index.as_query_engine()
        return search, len(docs)
    return None, 0

# Initialize memory system
search, session_count = load_index()

# Save
def save_convo(messages):
    global search, session_count
    
    with mem_lock:
        try:
            if os.path.exists(mem_file):
                with open(mem_file, "r") as file:
                    all_convos = json.load(file)
            else:
                all_convos = []
            
            all_convos.append(messages)
            
            with open(mem_file, "w") as file:
                json.dump(all_convos, file, indent=2)
            
            # Rebuild search index
            search, session_count = load_index()
            
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False

# Autosave when program exits
def save_on_exit():
    global convo
    if convo and len(convo) > 1:  # Only save actual conversations
        save_convo(convo)
        print("Session saved on exit.")

atexit.register(save_on_exit)

# Start a new conversation when mode is selected
def init_chat(mode, sys_prompt, msg_state):
    global convo, wait_feedback
    # Reset feedback state
    wait_feedback = False
    # Set system prompt based on mode
    if mode == "train":
        sys_prompt = main_prompt + training_mode
    else:
        sys_prompt = main_prompt
    # Initialize message state
    msg_state = [{"role": "system", "content": sys_prompt}]
    # Update current conversation for auto-saving
    convo = msg_state.copy()
    # Opening message
    greeting = "Hello! I'm your AI Medical Assistant.\nI'm here to support early detection of potential cognitive conditions.\nTo begin, please tell me your age and describe any symptoms you've been experiencing:"
    # Add greeting to messages
    msg_state.append({"role": "assistant", "content": greeting})
    convo.append({"role": "assistant", "content": greeting})
    
    return [[None, greeting]], sys_prompt, msg_state

# Main chat
def chat(message, history, mode, sys_prompt, msg_state):
    """Process messages and generate AI responses"""
    global convo, wait_feedback
    
    # Feedback
    if wait_feedback and mode == "train":
        if message.lower().startswith("yes"):
            # CORRECT
            feedback = "[Feedback] Early detection was correct."
            convo.append({"role": "user", "content": feedback})
            msg_state.append({"role": "user", "content": feedback})
            confirm = "Thank you for confirming. The early detection was recorded as correct. This case will help with future assessments."
            convo.append({"role": "assistant", "content": confirm})
            msg_state.append({"role": "assistant", "content": confirm})
            
            # Save convo w/ feedback
            save_convo(convo)
            wait_feedback = False
            
            return "", history + [[message, confirm]], sys_prompt, msg_state
            
            #INCORRECT
        elif message.lower().startswith("no"):
            parts = message.split(",", 1)
            correction = parts[1].strip() if len(parts) > 1 else "Unspecified"
            feedback = f"[Feedback] Early detection was incorrect. Correction: {correction}."
            convo.append({"role": "user", "content": feedback})
            msg_state.append({"role": "user", "content": feedback})
            confirm = f"Thank you for the correction. The early detection was updated to: {correction}. This will improve future assessments."
            convo.append({"role": "assistant", "content": confirm})
            msg_state.append({"role": "assistant", "content": confirm})
            
            # Save convo w/ feedback
            save_convo(convo)
            wait_feedback = False
            
            return "", history + [[message, confirm]], sys_prompt, msg_state
    
    # Init case
    if not msg_state:
        if mode == "train":
            sys_prompt = main_prompt + training_mode
        else:
            sys_prompt = main_prompt
            
        msg_state = [{"role": "system", "content": sys_prompt}]
        convo = msg_state.copy()
    
    # Update mode
    if mode == "train" and "TRAINING MODE" not in sys_prompt:
        sys_prompt = main_prompt + training_mode
        for i, msg in enumerate(msg_state):
            if msg["role"] == "system" and "You are a formal" in msg["content"]:
                msg_state[i]["content"] = sys_prompt
                convo[i]["content"] = sys_prompt
                break
    elif mode == "detect" and "TRAINING MODE" in sys_prompt:
        sys_prompt = main_prompt
        for i, msg in enumerate(msg_state):
            if msg["role"] == "system" and "You are a formal" in msg["content"]:
                msg_state[i]["content"] = sys_prompt
                convo[i]["content"] = sys_prompt
                break
    
    # Add user message
    msg_state.append({"role": "user", "content": message})
    convo.append({"role": "user", "content": message})
    
    # Search memory for similar cases
    if search:
        try:
            result = search.query(message)
            if result.response and len(result.response.strip()) > 10:
                # Instructions for using similar past case
                mem_prompt = (
                    "You previously encountered a similar user case with the following symptoms:\n\n"
                    f"\"\"\"{result.response.strip()}\"\"\"\n\n"
                    "When the current user's symptoms clearly match this past case, you MUST say:\n"
                    "\"These symptoms resemble a prior case I encountered.\"\n"
                    "Then explain the similarity and give your best early detection."
                )
                
                # Add memory to conversation context
                mem_added = False
                for i, msg in enumerate(msg_state):
                    if msg["role"] == "system" and "You previously encountered" in msg["content"]:
                        msg_state[i]["content"] = mem_prompt
                        mem_added = True
                        break
                
                if not mem_added:
                    msg_state.insert(1, {"role": "system", "content": mem_prompt})
        except Exception as e:
            print(f"Error searching memory: {e}")
    
    # Get AI response
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=msg_state,
            max_tokens=300
        )
        
        ai_msg = response.choices[0].message.content.strip()
        msg_state.append({"role": "assistant", "content": ai_msg})
        convo.append({"role": "assistant", "content": ai_msg})
        
        # Check for training completion
        if "[TRAINING COMPLETE]" in ai_msg.upper() and mode == "train":
            # Enable feedback collection
            wait_feedback = True
            
            # Add feedback instructions
            ai_msg += "\n\n[System: Is this early detection correct? Reply with 'Yes' or 'No, it's actually [condition]' if incorrect.]"
        
        # Clear input field
        return "", history + [[message, ai_msg]], sys_prompt, msg_state
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return "", history + [[message, error_msg]], sys_prompt, msg_state

# GUI
with gr.Blocks(title="Early Detection System") as app:
    sys_prompt = gr.State(main_prompt)
    msg_state = gr.State([])
    
    # Interface elements
    gr.Markdown("# Early Detection System")
    with gr.Row():
        mode = gr.Radio(
            ["detect", "train"], 
            label="Mode", 
            value=None  # Force user selection
        )
    
    chat_box = gr.Chatbot(
        [],
        elem_id="chatbot",
        height=500
    )
    
    with gr.Row():
        user_msg = gr.Textbox(
            placeholder="Type your message here...",
            scale=8
        )
        send = gr.Button("Send", scale=1)
    
    # Status display
    status = gr.Textbox(label="Chat History ", value=f"Ready. {session_count} past sessions available.")
    
    # Wire up interactions
    send.click(
        chat,
        inputs=[user_msg, chat_box, mode, sys_prompt, msg_state],
        outputs=[user_msg, chat_box, sys_prompt, msg_state]
    )
    
    user_msg.submit(
        chat,
        inputs=[user_msg, chat_box, mode, sys_prompt, msg_state],
        outputs=[user_msg, chat_box, sys_prompt, msg_state]
    )
    
    # Initialize chat when mode selected
    mode.change(
        lambda m: gr.update(),
        inputs=[mode],
        outputs=[]
    ).then(
        init_chat,
        inputs=[mode, sys_prompt, msg_state],
        outputs=[chat_box, sys_prompt, msg_state]
    )

# Start the app
if __name__ == "__main__":
    app.launch(share=False)