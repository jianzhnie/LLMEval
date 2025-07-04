import os
import time
import random
import openai
import logging
from packaging.version import parse as parse_version
from typing import Dict, Any

IS_OPENAI_V1 = parse_version(openai.__version__) >= parse_version("1.0.0")

if IS_OPENAI_V1:
    from openai import APIError, APIConnectionError, RateLimitError
else:
    from openai.error import APIError, APIConnectionError, RateLimitError


class ClientError(RuntimeError):
    """Custom exception class for client-related errors."""

    pass


def get_content(query: str, base_url: str, model_name: str) -> str:
    """
    Fetches content from the OpenAI API based on the provided query, base URL, and model name.

    Args:
        query (str): The user's input query.
        base_url (str): The base URL for the OpenAI API.
        model_name (str): The name of the model to use for generating the response.

    Returns:
        str: The generated content from the API.

    Raises:
        ClientError: If there are issues with the API request or response.
    """
    API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
    API_REQUEST_TIMEOUT = int(os.getenv("OPENAI_API_REQUEST_TIMEOUT", "99999"))

    if IS_OPENAI_V1:
        import httpx

        client = openai.OpenAI(
            api_key=API_KEY,
            base_url=base_url,
            timeout=httpx.Timeout(API_REQUEST_TIMEOUT),
        )
        call_func = client.chat.completions.create
        call_args = {
            "model": model_name,
            "messages": [{"role": "user", "content": query}],
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
            "extra_body": {},
            "timeout": API_REQUEST_TIMEOUT,
        }
        call_args["extra_body"].update({"top_k": 40})
    else:
        call_func = openai.ChatCompletion.create
        call_args = {
            "api_key": API_KEY,
            "api_base": base_url,
            "model": model_name,
            "messages": [{"role": "user", "content": query}],
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
            "request_timeout": API_REQUEST_TIMEOUT,
        }
        call_args.update({"top_k": 40})

    try:
        completion = call_func(**call_args)
        return completion.choices[0].message.content
    except AttributeError as e:
        err_msg = getattr(completion, "message", "")
        logging.warning(f"AttributeError encountered: {err_msg}")
        time.sleep(random.randint(25, 35))
        raise ClientError(err_msg) from e
    except (APIConnectionError, RateLimitError) as e:
        err_msg = e.message if IS_OPENAI_V1 else e.user_message
        logging.warning(f"API connection error or rate limit exceeded: {err_msg}")
        time.sleep(random.randint(25, 35))
        raise ClientError(err_msg) from e
    except APIError as e:
        err_msg = e.message if IS_OPENAI_V1 else e.user_message
        if "maximum context length" in err_msg:
            logging.warning(f"Maximum context length exceeded. Error: {err_msg}")
            return {"gen": "", "end_reason": "max length exceeded"}
        logging.warning(f"API error encountered: {err_msg}")
        time.sleep(1)
        raise ClientError(err_msg) from e


if __name__ == "__main__":
    conversation_history = []
    user_input = "Hello!"
    res = get_content(user_input, "http://10.77.249.36:8030/v1", "Qwen/QwQ")
    print(f"Response: {res}")

    user_input = "How are you?"
    res = get_content(user_input, "http://10.77.249.36:8030/v1", "Qwen/QwQ")
    print(f"Response: {res}")
