import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings

# Load API Key
load_dotenv("chatkey.env")
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# AI Instructions
system_prompt = """
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
- Actively observe the user’s language for vague phrasing, reduced vocabulary, word-finding pauses, broken sentence structure, or signs of disorganised thought for indications of cognitive decline.

Throughout the conversation:
- Accumulate information over time. Combine evidence from multiple domains (language, motor, behavioural, memory) to identify meaningful patterns.
- Once a consistent pattern of symptoms is observed, you MUST determine the most likely neurodegenerative disease.
- Do NOT vaguely say “an issue” or “a condition.” You MUST name a specific disease (e.g., Parkinson’s disease).
- Use your understanding of symptom patterns to reason your conclusion, and then clearly state it.
"""

# Chat mode
mem_file = "chatmem.json"
mode = input("Enter mode (train / detect): ").lower().strip()

if mode == "train":
    system_prompt += """
    
TRAINING MODE
If you decide to suggest a likely neurodegenerative disease based on the user's symptoms, you must end your message with the exact phrase: [TRAINING COMPLETE]. This signals that feedback will follow.
"""

# Setup embedding model
Settings.embed_model = OpenAIEmbedding()

# Parse full sessions
def parse_chat_memory(chatmem_path):
    if not os.path.exists(chatmem_path):
        return []
    with open(chatmem_path, "r") as file:
        sessions = json.load(file)
    documents = []
    for session in sessions:
        text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in session])
        documents.append(Document(text=text))
    return documents

# Index memory
documents = parse_chat_memory(mem_file)
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Start conversation
messages = [{"role": "system", "content": system_prompt}]

# Initial input
user_input = input("Hello! I'm your AI Medical Assistant.\n"
                   "I'm here to support early detection of potential cognitive conditions.\n"
                   "To begin, please tell me your age and describe any symptoms you've been experiencing:\n\nYou: ")
messages.append({"role": "user", "content": user_input})

# Begin interactive loop
while True:
    try:
        # Retrieve matching memory before GPT responds
        retrieved = query_engine.query(user_input)
        if retrieved.response:
            memory_injection = (
                "You previously encountered a similar user case with the following symptoms:\n\n"
                f"\"\"\"{retrieved.response.strip()}\"\"\"\n\n"
                "When the current user's symptoms clearly match this past case, you MUST say:\n"
                "\"These symptoms resemble a prior case I encountered.\"\n"
                "Then explain the similarity and give your best early detection."
            )
            messages.insert(1, {"role": "system", "content": memory_injection})

        # Get response from GPT
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=300
        )

        reply = response.choices[0].message.content.strip()
        print(f"\nAI: {reply}")
        messages.append({"role": "assistant", "content": reply})

        # Feedback logic for training
        if "[training complete]" in reply.lower() and mode == "train":
            confirmed = input("\nWas suggested early detection correct? (yes/no): ").strip().lower()
            feedback = {"correct": confirmed}
            if confirmed == "no":
                corrected = input("Please provide the correct early detection: ").strip()
                feedback["correct_diagnosis"] = corrected
            messages.append({
                "role": "user",
                "content": f"[Feedback] Prediction was {confirmed}. Correction: {feedback.get('correct_diagnosis', 'N/A')}."
            })

        # Get next user input
        user_input = input("\nYou (or type 'exit' to end): ")
        if user_input.lower() in ["exit", "quit"]:
            print("\nThank you for your time. Stay safe!")
            break
        messages.append({"role": "user", "content": user_input})

    except Exception as e:
        print("ERROR:", e)
        break

# Save session
if os.path.exists(mem_file):
    with open(mem_file, "r") as file:
        all_sessions = json.load(file)
else:
    all_sessions = []

all_sessions.append(messages)

with open(mem_file, "w") as file:
    json.dump(all_sessions, file, indent=2)
