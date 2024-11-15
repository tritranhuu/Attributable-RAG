def format_doc_simple(doc):
    template = f"""{doc}"""
    return template

def format_doc(i, doc):
    template = f"""[{i}]: {doc}"""
    return template

def get_post_cite_input(query, docs):
    template = (
        "Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone."
        "\n\n"
        "Search Results: \n"
        "{context}"
    )
    context = "\n-----------\n".join([format_doc_simple(c["segment"]) for i, c in enumerate(docs)])
    return template.format(context=context.strip()), query


def get_icl_input_llama(query, docs):
    template = """You're a helpful AI assistant. Your mission is to give a full and complete answer for the question. \
Cite each context document inline that supports your answer within brackets [] using the IEEE format. \
Ensure each sentence is properly cited.

Context:
{context}
"""
    context = "".join([format_doc(i, c["segment"]) for i, c in enumerate(docs)])
    return template.format(context=context.strip()), query

def get_icl_input_mistral(query, docs):
    template = (
        "Forget everything you know about the world, this is what you know:"
        "\n{context}"
        "Now, write an answer for this question:\n"
        "{query}"
        "If you don't know the answer, say that you "
        "don't know. Your answer needs to be in an accurate, engaging, concise and well structured paragraph."
        "\nNOTE: Cite at least one document where you found the information in the end of each sentence. "
        # "\nNOTE: Include the DocIDs as the ciations of the documents where you found the information for the answer in EVERY sentence, "
        "When citing use square bracket [docid_1][docid_2]... , for excample [1][2]."
    )
    context = "".join([format_doc(i, c["segment"]) for i, c in enumerate(docs)])
    return "", template.format(context=context.strip(), query=query)


def get_finetuned_input_llama(query, docs):
    template = """You're a helpful AI assistant. Your mission is to give a full and complete answer for the question. \
Cite each context document inline that supports your answer within brackets [] using the IEEE format. \
Ensure each sentence is properly cited.

Context:
{context}
"""
    context = "".join([format_doc(i, c["segment"]) for i, c in enumerate(docs)])
    return template.format(context=context.strip()), query

def get_finetuned_input_mistral(query, docs):
    template = """You're a helpful AI assistant. Your mission is to give a full and complete answer for the question. \
Cite each context document inline that supports your answer within brackets [] using the IEEE format. \
Ensure each sentence is properly cited.

Context:
{context}

Question: {query}
"""
    context = "".join([format_doc(i, c["segment"]) for i, c in enumerate(docs)])
    return "", template.format(context=context.strip(), query=query)