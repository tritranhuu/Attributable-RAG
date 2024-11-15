import transformers
import torch

class QueryOptimizer:
    def __init__(self):
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="cuda",
        )

    def rewrite_query(self, query):
        messages = [
            {"role": "system",
             "content": "Provide ONE better search query for web search engine to answer the given question. Just re-write the query and do not say anything else"},
            {"role": "user", "content": query},
        ]

        outputs = self, pipeline(
            messages,
            max_new_tokens=256,
            temperature=0.9,
            num_beams=1
        )
        return outputs[0]["generated_text"][-1]['content'].strip("\"")

    def decompse_query(self, query):
        system = """You are an expert at converting user questions into database queries. \

        Perform query decomposition. Given a user question, break it down into distinct sub questions that \
        you need to answer in order to answer the original question.

        If there are acronyms or words you are not familiar with, do not try to rephrase them.
        Just return the sub questions, seperated by \n and do not say anything else"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query}
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=256,
            temperature=0.9,
            num_beams=3
        )
        return outputs[0]["generated_text"][-1]['content'].strip("\"").split("\n")

if __name__ == '__main__':
    qo = QueryOptimizer()
    query = "directv royal oak mi"
    print(qo.rewrite_query(query))