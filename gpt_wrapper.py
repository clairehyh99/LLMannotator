from openai import OpenAI

client = OpenAI()  # Automatically reads OPENAI_API_KEY


def _to_text(x):
    """
    Convert objects passed from R/reticulate into a clean text string.

    Behavior:
    ---------
    - If `x` is a list or tuple:
        Convert each element to string and join with newline characters.
    - Otherwise:
        Convert directly using str().

    This ensures that any structured object becomes a single text block
    suitable for prompting LLMs.
    """
    if isinstance(x, (list, tuple)):
        return "\n".join(str(xx) for xx in x)
    return str(x)


def run_gpt(prompt, desc, model="gpt-5.1"):
    """
    Send a system instruction + user description to an OpenAI model.

    Parameters
    ----------
    prompt : str or list
        System instruction block (e.g. "baseprompt + kable rules").
        Can be a string or a list of strings from R.
    
    desc : str or list
        Description of the current item, such as:
            '1: Photosynthesis rate from Licor 6400'

    model : str
        The model to call (default: 'gpt-5.1').

    What the function does:
    -----------------------
    1. Standardizes both `prompt` and `desc` into text using `_to_text()`.
    2. Concatenates them into a structured multi-section prompt:
           System instruction:
           ...
           
           User description:
           ...
    3. Sends this to the OpenAI Responses API.
    4. Returns the generated text.
       If parsing fails, return the raw response for debugging.

    Returns
    -------
    str
        The model's text output. If extraction fails, the raw response is
        returned instead.
    """

    prompt_text = _to_text(prompt)
    desc_text   = _to_text(desc)

    full_input = (
        "System instruction:\n"
        + prompt_text
        + "\n\n"
        + "User description:\n"
        + desc_text
    )

    resp = client.responses.create(
        model=model,
        input=full_input,
        max_output_tokens=4000,
        temperature=0
    )

    try:
        # Standard extraction path
        return resp.output[0].content[0].text
    except Exception:
        # Fallback for debugging
        return str(resp)
