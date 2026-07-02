import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Dynamic LLM generation helper
def get_llm(config: RunnableConfig):
    configurable = config.get("configurable", {})
    model_name = configurable.get("model_name", "llama-3.3-70b-versatile")
    temperature = configurable.get("temperature", 0.3)
    
    return ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        model=model_name,
        temperature=temperature
    )

# Reconfigure stdout to use UTF-8 on Windows to prevent emoji encoding crashes
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add this import to fix the error!
from dotenv import load_dotenv

# Load environment variables (your .env file)
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Using Groq's insanely fast inference
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

# ==========================================
# 2. AGENTIC ROUTER & LANGGRAPH SETUP
# ==========================================
class AgentWorkspaceState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    research_data: str
    final_artifact: str
    route: str # Tracks which path the graph should take

def router_agent(state: AgentWorkspaceState, config: RunnableConfig):
    """The 'Brain' that decides if the user wants to chat or needs a report."""
    user_input = state['messages'][-1].content
    local_llm = get_llm(config)
    
    # We ask the LLM to classify the user's intent
    prompt = f"""You are an intelligent routing agent.
    If the user is greeting you, asking a simple question, or requesting something casual (like a recipe, a joke, or general knowledge), output the word: CHAT
    If the user is explicitly asking for a deep analysis, a comprehensive report, or complex research, output the word: RESEARCH
    
    User Input: "{user_input}"
    Respond with ONLY the word CHAT or RESEARCH."""
    
    response = local_llm.invoke(prompt).content.strip().upper()
    
    # Clean up the response just in case the LLM is chatty
    route = "RESEARCH" if "RESEARCH" in response else "CHAT"
    print(f"🧠 Router decided to go to: {route}")
    
    return {"route": route}

def chat_agent(state: AgentWorkspaceState, config: RunnableConfig):
    """Handles standard, conversational chatbot requests seamlessly."""
    print("💬 Chat Agent is responding naturally...")
    local_llm = get_llm(config)
    
    configurable = config.get("configurable", {})
    custom_system_prompt = configurable.get("system_prompt", "").strip()
    
    system_text = custom_system_prompt if custom_system_prompt else "You are a helpful, brilliant, and friendly AI assistant. Provide clear, well-formatted answers."
    sys_msg = SystemMessage(content=system_text)
    messages = [sys_msg] + state['messages']
    
    response = local_llm.invoke(messages)
    return {"final_artifact": response.content, "messages": [response]}

def researcher_agent(state: AgentWorkspaceState, config: RunnableConfig):
    """Gathers complex data for full reports."""
    print("🤖 Researcher is gathering deep analysis data...")
    local_llm = get_llm(config)
    
    prompt = f"Provide a detailed, factual breakdown regarding this topic to be used for a report: {state['messages'][-1].content}"
    response = local_llm.invoke(prompt)
    return {"research_data": response.content}

def writer_agent(state: AgentWorkspaceState, config: RunnableConfig):
    """Drafts the formal report."""
    print("✍️ Writer is drafting the formal report...")
    local_llm = get_llm(config)
    
    prompt = f"Using this research: {state['research_data']}, write a comprehensive and highly professional markdown report. Include headers and bullet points."
    response = local_llm.invoke(prompt)
    return {"final_artifact": response.content, "messages": [response]}

# --- Build the Graph ---
workflow = StateGraph(AgentWorkspaceState)

# Add all our nodes
workflow.add_node("router", router_agent)
workflow.add_node("chat", chat_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)

# The entry point is ALWAYS the router
workflow.set_entry_point("router")

# The router uses a conditional edge to decide where to go
def route_next(state: AgentWorkspaceState):
    if state["route"] == "RESEARCH":
        return "researcher"
    return "chat"

workflow.add_conditional_edges("router", route_next)
workflow.add_edge("chat", END) 
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END) 

# 1. Add memory checkpointer to the compiled graph
memory = MemorySaver()
agent_app = workflow.compile(checkpointer=memory)

# ==========================================
# 3. FASTAPI ENDPOINT
# ==========================================
@app.get("/")
async def get_index():
    return FileResponse("index.html")

class SettingsSchema(BaseModel):
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.3
    system_prompt: str = ""

class UserRequest(BaseModel):
    prompt: str
    session_id: str
    settings: SettingsSchema = None

class TitleRequest(BaseModel):
    prompt: str

@app.post("/api/generate-title")
async def generate_title(req: TitleRequest):
    prompt = f"Generate a short, catchy 2-to-4 word title representing this search/chat query. For simple greetings (e.g. 'hello', 'hi', 'hey'), respond with a simple title like 'Greeting' or 'Casual Chat'. Respond with ONLY the title words, no quotes, no markdown, no punctuation:\n\nQuery: {req.prompt}"
    response = llm.invoke(prompt)
    title = response.content.strip().replace('"', '').replace("'", "").replace(".", "").strip()
    if len(title) > 30:
        title = title[:30] + "..."
    return {"title": title}

@app.post("/api/run-agents")
async def run_agents(req: UserRequest):
    model_name = "llama-3.3-70b-versatile"
    temperature = 0.3
    system_prompt = ""
    
    if req.settings:
        model_name = req.settings.model
        temperature = req.settings.temperature
        system_prompt = req.settings.system_prompt

    # 2. We use the session_id to maintain a continuous thread/memory!
    config = {
        "configurable": {
            "thread_id": req.session_id,
            "model_name": model_name,
            "temperature": temperature,
            "system_prompt": system_prompt
        }
    }
    
    # 3. We only pass the NEW message. Our state logic automatically appends it to history.
    input_state = {
        "messages": [HumanMessage(content=req.prompt)]
    }
    
    final_state = agent_app.invoke(input_state, config=config)
    return {"artifact": final_state["final_artifact"], "route": final_state.get("route", "CHAT")}