# ICL Unstructured Prompt
def format_doc_icl_up(i, doc):
    template = f"""[{i}]: {doc}"""
    return template


def build_prompt_icl_up(candidates, query):
    template = (
        "Given the following contexts:"
        "\n{context}"
        "Now, write an answer for this question based on the provied context:\n"
        "{query}"
        "If you cannot fine the answer in the contexts, say that you "
        "don't know. Your answer needs to be in an accurate, engaging, concise and well structured paragraph."
        "\nNOTE: Cite at least one document where you found the information in the end of each sentence. "
        "When citing use square bracket [docid_1][docid_2]... , for excample [1][2]."
    )

    if type(candidates[0]) == dict:
        context = "\n".join([format_doc_icl_up(i, c["segment"]) for i, c in enumerate(candidates)])
    else:
        context = "\n".join([format_doc_icl_up(i, c) for i, c in enumerate(candidates)])

    messages = [
        {"role": "user", "content": template.format(context=context.strip(), query=query)},
    ]

    return messages

# ICL Structured Prompt
def format_doc_icl_sp(i, doc):
    template = f"""\
<document>
        <doc_id>{i}</doc_id>
        <text>{doc}</text>
    </document>
    """

    return template


# If none of the articles answer the question, just say you don't know.

def build_prompt_icl_sp(candidates, query):
    template = """You're a helpful AI assistant. Given a user question and some article snippets, \
answer the user question and provide citations.

Remember, you must return both an answer and citations. A citation consists of a VERBATIM quote that \
justifies the answer and the IDs of the quote articles. Return all citations for every quote across all articles \
that justify the answer. Use the following format for your final output:

<cited_answer>
    <statement>
        <citations>
            <doc_id></docid>
            ...
        </citations>
        <text></text>
    </statement>
    ...
</cited_answer>

Here are the articles:
<documents>
    {context}
</documents>

Question: {query}"""
    if type(candidates[0]) == dict:
        context = "\n".join([format_doc_icl_sp(i, c["segment"]) for i, c in enumerate(candidates)])
    else:
        context = "\n".join([format_doc_icl_sp(i, c) for i, c in enumerate(candidates)])

    messages = [
        {"role": "user", "content": template.format(context=context.strip(), query=query)},
    ]
    return messages

#FT-Llama
def format_doc_ft_llama(i, doc):
    template = f"""[{i}]: {doc}"""
    return template


def build_prompt_ft_llama(candidates, query):
    template = """You're a helpful AI assistant. Your mission is to give a full and complete answer to the question. \
Cite each context document inline that supports your answer within brackets [] using the IEEE format. \
Ensure each sentence is properly cited.

Context:
{context}
"""
    if type(candidates[0]) == dict:
        context = "\n".join([format_doc_ft_llama(i, c["segment"]) for i, c in enumerate(candidates)])
    else:
        context = "\n".join([format_doc_ft_llama(i, c) for i, c in enumerate(candidates)])
    messages = [
        {"role": "system", "content": template.format(context=context.strip())},
        {"role": "user", "content": query},
    ]

    return messages

#FT-Mistral
def format_doc_ft_mistral(i, doc):
    template = f"""[{i}]: {doc}"""
    return template

def build_prompt_ft_mistral(candidates, query):
    template = """You're a helpful AI assistant. Your mission is to give a full and complete answer to the question. \
Cite each context document inline that supports your answer within brackets [] using the IEEE format. \
Ensure each sentence is properly cited.

Context:
{context}

Question: {query}
"""
    if type(candidates[0]) == dict:
        context = "\n".join([format_doc_ft_mistral(i, c["segment"]) for i, c in enumerate(candidates)])
    else:
        context = "\n".join([format_doc_ft_mistral(i, c) for i, c in enumerate(candidates)])
    messages = [
        {"role": "user", "content": template.format(context=context.strip(), query=query)}
    ]

    return messages

# FT-Flan-T5
def format_doc_ft_t5(i, doc):
    template = f"""[{i}]: {doc}"""
    return template


def build_prompt_ft_t5(candidates, query):
    template = """You're a helpful AI assistant. Your mission is to give a full and complete answer to the question. \
Cite each context document inline that supports your answer within brac/QAkets [] using the IEEE format. \
Ensure each sentence is properly cited._icl

Context:
{context}

Question: {query}
"""
    if type(candidates[0]) == dict:
        context = "\n".join([format_doc_ft_t5(i, c["segment"]) for i, c in enumerate(candidates)])
    else:
        context = "\n".join([format_doc_ft_t5(i, c) for i, c in enumerate(candidates)])
    messages = [
        {"role": "user", "content": template.format(context=context.strip(), query=query)}
    ]
    return messages

# PG-LLama

# PG-Mistral

PROMPT_BUILDER_DICT = {
    "icl-llama-up": build_prompt_icl_up,
    "icl-mistral-up": build_prompt_icl_up,
    "icl-llama-sp": build_prompt_icl_sp,
    "icl-mistral-sp": build_prompt_icl_sp,
    "ft-llama": build_prompt_ft_llama,
    "ft-mistral": build_prompt_ft_mistral,
    "ft-t5": build_prompt_ft_t5,
}