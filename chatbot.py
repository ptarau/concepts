import os
import openai
from config import CF


def get_model() -> str:
    if CF.USE_OLLAMA:
        return CF.OLLAMA_MODEL
    else:
        return CF.GPT_MODEL


def get_llm_name() -> str:
    if CF.USE_OLLAMA:
        return "ollama"
    else:
        return "gpt"


def get_client() -> openai.OpenAI:
    if CF.USE_OLLAMA:
        # Ollama API is OpenAI-compatible (does not check API key)
        return openai.OpenAI(
            base_url=CF.OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama ignores the key, but requires it
        )
    else:
        api_key = CF.API_KEY
        return openai.OpenAI(api_key=api_key)


def get_cost_rates():
    model = CF.GPT_MODEL
    if model == "gpt-5":
        input_rate = 1.25  # per 1m input tokens
        output_rate = 10.00  # per 1m output tokens
    elif model == "gpt-5-mini":
        input_rate = 0.25  # per 1m input tokens
        output_rate = 2.00  # per 1m output tokens
    elif model == "gpt-5-nano":
        input_rate = 0.05  # per 1m input tokens
        output_rate = 0.40  # per 1m output tokens
    else:
        input_rate, output_rate = 0, 0
    return input_rate / 1000000, output_rate / 1000000


def ask(prompt: str) -> tuple[str, float]:
    """Return the response from the OpenAI API for a given prompt."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
        )
        if response is None:
            print("*** No response from OpenAI API")
            exit(1)

    except openai.OpenAIError as e:

        print("*** OpenAIError:", e)
        print("LLM:", get_llm_name())
        print("LLM model:", get_model())
        raise ConnectionRefusedError(get_llm_name() + "-->" + get_model())

    if CF.USE_OLLAMA:
        cost = 0.0
    elif response.usage is None:
        cost = 0.0
    else:
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        input_rate, output_rate = get_cost_rates()

        cost = (input_tokens * input_rate) + (output_tokens * output_rate)

    answer = response.choices[0].message.content
    if answer is None:
        print("*** NO ANSWER from OpenAI!")
        return "", cost
    else:
        return answer, cost


if __name__ == "__main__":
    prompt = "What is the capital of France?"
    answer, cost = ask(prompt)
    print("Answer:", answer)
    print("Cost: $%.8f" % cost)
