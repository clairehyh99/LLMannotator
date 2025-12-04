from openai import OpenAI

client = OpenAI()  # auto read OPENAI_API_KEY

def _to_text(x):
    """
    把 R/reticulate 传过来的各种类型，统一变成一个长字符串。
    - 如果是 list / tuple：逐个转成 str，再用换行拼起来
    - 其他类型：直接 str()
    """
    if isinstance(x, (list, tuple)):
        return "\n".join(str(xx) for xx in x)
    return str(x)

def run_gpt(prompt, desc, model="gpt-5.1"):
    """
    prompt: system 指令（比如 baseprompt + kable 那一大段）
    desc:   当前这条描述，比如 '1: Photosynthesis rate from Licor 6400'
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
        return resp.output[0].content[0].text
    except Exception:
        return str(resp)
