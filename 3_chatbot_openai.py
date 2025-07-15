from typing import Annotated
from dotenv import load_dotenv
load_dotenv()

from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Initialize OpenAI chat model
llm = ChatOpenAI(model="gpt-3.5-turbo")

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("chatbot_node", chatbot)

builder.add_edge(START, "chatbot_node")
builder.add_edge("chatbot_node", END)

graph = builder.compile()

# Test the chatbot
if __name__ == "__main__":
    # Test with a simple message
    message = {"role": "user", "content": "Who walked on the moon for the first time? Print only the name"}
    response = graph.invoke({"messages": [message]})
    print("Test response:", response["messages"][-1].content)

    # Interactive chat loop
    print("\nStarting interactive chat (type 'quit' or 'exit' to end):")
    state = None
    while True:
        in_message = input("You: ")
        if in_message.lower() in {"quit", "exit"}:
            break
        if state is None:
            state: State = {
                "messages": [{"role": "user", "content": in_message}]
            }
        else:
            state["messages"].append({"role": "user", "content": in_message})

        state = graph.invoke(state)
        print("Bot:", state["messages"][-1].content)
