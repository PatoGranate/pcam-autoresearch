import anthropic
from tools import tools, run_tool

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 4096
client = anthropic.Anthropic()


def call_model(messages):
    """ Send message history to model and ask it for a response or tool use

    Args:
        messages (list) : list of {sender: content} where sender could be either user or assistant, and content is the 
        content of the message.

    Returns:
        response (SimpleNamespace) : stop_reason, content. Stop_reason gives information about the following content.
    """
    response = client.messages.create(
        model = MODEL,
        max_tokens = MAX_TOKENS,
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
    response = client.messages.create(
    model = MODEL,
    max_tokens = MAX_TOKENS,
    tools = tools,
    tool_choice = {"type": "none"},
    messages = messages,
    )
    return response


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

    # Set a max limit of turns to prevent agent from using tools infinitely, and initialize an error counter
    max_turns = 10
    error_counter = 0
    
    for i in range(max_turns):
        # For each turn, call the model, retrieve its response, and extract the stop_reason and content
        response = call_model(messages)
        stop_reason = response.stop_reason
        content = response.content

        # If end turn (no more tools requested), return the final output text from the model
        if stop_reason == "end_turn":
            text = next((block for block in content if block.type == "text"), None)
            if text is None:
                return "Agent stopped without a final answer"
            return text.text

        # If a tool was requested, run the tool in tools.py
        elif stop_reason == "tool_use":
            tool_use = next(block for block in content if block.type == "tool_use")
            tool_result, is_error = run_tool(tool_use.name, tool_use.input)

        # If another stop_reason is recorded, break with error message
        else:
            return f"Unexpected stop reason: {stop_reason}"

        if is_error:
            error_counter += 1
        else:
            error_counter = 0

        # Save both the assistant and user output messages to input as context for the next call
        messages.append({"role": "assistant",
                            "content": content},)
        messages.append({"role": "user",
                            "content": [{"type": "tool_result", 
                                        "is_error": is_error,
                                        "tool_use_id": tool_use.id, 
                                        "content": tool_result}]})

        # If three errors in a row, cut the agent from the loop
        if error_counter >= 3:
            print("Error counter maxed out")
            break

        if i == 9:
            print("Agent ran out of turns")

    # Final text only call
    messages.append({"role": "user",
                     "content": "You are not able to use any more tools. Summarise what you found so far in your last text only call"})

    final_response = call_model_no_tools(messages)
    content = final_response.content
    text = next((block for block in content if block.type == "text"), None)
    if text is None:
        return "Agent stopped without a final answer"
    return text.text

if __name__ == "__main__":
    initial_message = "Try to read a few files in this repo, come back with what you find"
    output = run_agent(initial_message)
    print(output)