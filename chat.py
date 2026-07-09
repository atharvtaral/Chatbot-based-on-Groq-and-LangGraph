import os
import gradio as gr
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

# --- 1. LangGraph Setup (Your original code) ---
load_dotenv()

llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq")

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot_node(state: State) -> Dict[str, Any]:
    messages = state.get("messages", [])
    response = llm.invoke(messages)
    return {"messages": [response]}

def samplenode(state: State):
    return {"messages": [AIMessage(content="[Sample Node: Process Complete]")]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("samplenode", samplenode)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "samplenode")
graph_builder.add_edge("samplenode", END)
graph = graph_builder.compile()


# --- 2. GRADIO UI Setup ---
def predict(message, history):
    # 1. Convert the existing history into LangGraph format.
    graph_input_messages = []
    for human, ai in history:
        if human: graph_input_messages.append(HumanMessage(content=human))
        if ai: graph_input_messages.append(AIMessage(content=ai))
    
    # 2. Add a new message
    graph_input_messages.append(HumanMessage(content=message))
    
    # 3. Run LangGraph
    output_state = graph.invoke({"messages": graph_input_messages})
    
    # 4. Display the last message from the LLM or your node on the UI.
    last_msg = output_state["messages"][-2] # Chatbot node message (second to last, because 'samplenode' runs after it)
    
    return last_msg.content

# Launch Gradio ChatInterface
demo = gr.ChatInterface(
    fn=predict, 
    title="LangGraph Chatbot",
    description="Chatbot based on Groq and LangGraph"
)

if __name__ == "__main__":
    demo.launch(share=False) # To run locally. Set share=True if you need an internet link.