import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from openai import OpenAI

app = FastAPI()

# Allows your website to talk to this code
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- API CLIENTS (Set these in your hosting environment) ---
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_KEY)
client = OpenAI(api_key=OPENAI_KEY)
search_tool = TavilySearchResults(k=7, api_key=TAVILY_KEY)

class ChatRequest(BaseModel):
    query: str
    user_id: str

@app.post("/api/chat")
async def aetheris_logic(request: ChatRequest):
    user_input = request.query
    
    # 1. DEEP SEARCH
    search_data = search_tool.run(user_input)

    # 2. INITIAL DRAFT
    prompt = f"System: You are AETHERIS. Use this data: {search_data}\n\nUser: {user_input}"
    current_answer = llm.predict(prompt)

    # 3. THE 4X SELF-CORRECTION LOOP
    for i in range(3): 
        check_prompt = f"Check this for mistakes or shallow info: {current_answer}. If it is perfect, say 'PASS'. If not, rewrite it better."
        critique = llm.predict(check_prompt)
        if "PASS" in critique:
            break
        current_answer = critique

    # 4. IMAGE GENERATION (Triggered by keywords)
    img_url = None
    if any(word in user_input.lower() for word in ["create", "generate", "show", "image"]):
        gen = client.images.generate(model="dall-e-3", prompt=f"Futuristic 4k: {user_input}")
        img_url = gen.data[0].url

    return {"answer": current_answer, "image_url": img_url, "status": "Verified 4x Logic"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
