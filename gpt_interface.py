from openai import OpenAI

def run_gpt_completion(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """
    Runs a GPT-4 chat completion with a system and user prompt.

    Args:
        client (OpenAI): An instance of the OpenAI client.
        system_prompt (str): System-level instructions for the assistant.
        user_prompt (str): The user's message to the assistant.

    Returns:
        str: The assistant's generated reply.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    response_message = response.choices[0].message.content
    assert isinstance(response_message, str), "GPT response message should never be None."
    return response_message
