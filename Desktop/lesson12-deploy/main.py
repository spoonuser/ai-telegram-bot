import os
import json
import numpy as np
import redis
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not OPENAI_KEY:
    raise Exception("OPENAI_KEY not found")
if not TELEGRAM_TOKEN:
    raise Exception("TELEGRAM_TOKEN not found")

client = OpenAI(api_key=OPENAI_KEY)

redis_client = redis.from_url(
    os.getenv("REDIS_URL"),
    decode_responses=True
)

def get_user_messages(user_id):
    data = redis_client.get(str(user_id))
    
    if data:
        return json.loads(data)
    else:
        messages = [
            {"role":"system", "content":"You are a helpful AI assistant."}
        ]
        redis_client.set(str(user_id), json.dumps(messages))
        return messages
    
def save_user_messages(user_id, messages):
    redis_client.set(str(user_id), json.dumps(messages))
    
with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
chunks = text.split("\n")
valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def preprocess(text):
    return text.lower()
chunk_embeddings = [
    get_embedding(preprocess(chunk))
    for chunk in valid_chunks
]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_relevant_chunks(query, top_k = 3):
    query_embedding = get_embedding(preprocess(query))
    
    scores = []
    for i, emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((score, valid_chunks[i]))
    
    scores.sort(reverse=True)
    return [chunk for _, chunk in scores[:top_k]]

rag_intent = """
oil change price
server cost
car repair price
bmw repair
toyota repair
diagnostics cost
location almaty working hours
"""

rag_intent_embedding = get_embedding(preprocess(rag_intent))

def is_rag_query(query, threshold=0.4):
    query_embedding = get_embedding(preprocess(query))
    score = cosine_similarity(query_embedding, rag_intent_embedding)
    print("DEBUG SCORE:", score)
    return score > threshold

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_input = update.message.text
    
    messages = get_user_messages(user_id)
    user_rag = is_rag_query(user_input)
    
    if user_rag:
        relevant_chunks = find_relevant_chunks(user_input)
        context_text = "\n".join(relevant_chunks)
        
        print("DEBUG CONTEXT:", context_text)
        
        prompt = f"""
You MUST answer using ONLY the context below.

Context:
{context_text}

Question:
{user_input}
If answer is not in context say "I don't know."
"""
    else:
        prompt = user_input
        
    messages.append({
        "role":"user",
        "content": prompt
    })
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    answer = response.choices[0].message.content
    
    messages.append({
    "role":"assistant",
    "content": answer
    })
    
    if len(messages) > 12:
        messages = [messages[0]] + messages[-10:]
        
    save_user_messages(user_id, messages)
        
    await update.message.reply_text(answer)

app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

print("Bot is running...")
app.run_polling()


