import anthropic
from types import SimpleNamespace

from tools import tools, run_tool

def call_model(messages):
    """ Send message history to model and ask it for a response or tool use

    Args:
        messages (list) : list of {sender: content} where sender could be either user or assistant, and content is the 
        content of the message.

    Returns:
        response (SimpleNamespace) : stop_reason, content. Stop_reason gives information about the following content.
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model = MODEL,
        max_tokens = 1024,
        tools = tools,
        tool_choice = {"type": "auto", "disable_parallel_tool_use": True},
        messages = messages,
    )
    return response


def call_model_no_tools(messages):
    """ Send message history to model and ask only for a text response
     
    Args:
        messages (list) : list of {sender: content} where sender could be either user or assistant, and content is the 
        content of the message.
    
    Returns:
        response (SimpleNamespace) : stop_reason, content. Stop_reason gives information about the following content. In this
        case stop_reason should only be end_turn.
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
    model = MODEL,
    max_tokens = 1024,
    tools = tools,
    tool_choice = {"type": "none"},
    messages = messages,
    )
    return response


def call_model_fake(messages, turn):
    """temporary test in case there is no API key"""
    if turn == 1:
        stop_reason = "tool_use"
        content = [
            SimpleNamespace(type="text", text="I'll look at the configs folder."),
            SimpleNamespace(type="tool_use", id="toolu_01ABC", name="list_files", input={"path_str": "configs"}),
        ]
    elif turn == 2:
        stop_reason = "end_turn"
        content = [
            SimpleNamespace(type="text", text="The configs folder contains baseline.yaml, resnet18.yaml and some subfolders."),
        ]

    return SimpleNamespace(stop_reason = stop_reason, content = content)


def run_agent(goal):
    """ Run the agent in a loop by giving it a goal and receiving the agent's response

    Args:
        goal (str) : User's first message, asking the agent to do something
    
    Returns:
        text.text (str) : Model's final text only output after having access to tools.
    """
    # Set the goal as a message that call_model can accept as input
    messages = [
        {
            "role": "user",
            "content": goal
        }
    ]

    # Set a max limit of turns to prevent agent from using tools infinitely
    max_turns = 10

    
    for i in range(max_turns):
        # For each turn, call the model, retrieve its response, and extract the stop_reason and content
        response = call_model(messages)
        stop_reason = response.stop_reason
        content = response.content

        # If end turn (no more tools requested), return the final output text from the model
        if stop_reason == "end_turn":
            text = next(block for block in content if block.type == "text")
            return text.text

        # If a tool was requested, run the tool in tools.py
        elif stop_reason == "tool_use":
            tool_use = next(block for block in content if block.type == "tool_use")
            tool_result = run_tool(tool_use.name, tool_use.input)

        # If another stop_reason is recorded, break with error message
        else:
            return f"Unexpected stop reason: {stop_reason}"

        # Save both the assistant and user output messages to input as context for the next call
        messages.append({"role": "assistant",
                         "content": content},)
        messages.append({"role": "user",
                         "content": [{"type": "tool_result", 
                                      "tool_use_id": tool_use.id, 
                                      "content": tool_result}]})

        # If any error using tools, break the loop and make a single text only call.
        if tool_result.startswith("Error:"):
            break

    final_response = call_model_no_tools(messages)
    content = final_response.content
    text = next(block for block in content if block.type == "text")
    return text.text


MODEL = "claude-haiku-5-5"

if __name__ == "__main__":
    initial_message = "Look at the configs folder, and tell me what it contains."
    output = run_agent(initial_message)
    print(output)