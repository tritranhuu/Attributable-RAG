import gradio as gr
from pathlib import Path
from elasticsearch import Elasticsearch, helpers, RequestsHttpConnection
import json
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer,
                          DPRReader, DPRReaderTokenizer)
from threading import Thread
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from rerankers import Reranker
from DiskVectorIndex import DiskVectorIndex

from .prompt_utils import *
import re
import os

stop_words = set(nltk.corpus.stopwords.words('english'))

os.environ['COHERE_API_KEY'] = ""
index = DiskVectorIndex("Cohere/trec-rag-2024-index")

css_path = Path('/nfs/primary/RAG_Demo/data/frontend/css/index.css')

"""ES Retriever"""

es_conn = Elasticsearch([{"host": "172.30.104.59", "port": 9200}], timeout=50)
es_conn.ping()


def retrieval_candidates(
        es_conn: Elasticsearch,
        query,
        top_k=10
):
    should_clauses = [{
        'match': {
            "segment": {
                "query": query,
            }
        }
    }]
    query = {
        'query': {
            'bool': {
                'should': should_clauses
            }
        },
        "from": 0,
        "size": top_k,
    }
    res = es_conn.search(index="ms_marco_21", body=json.dumps(query))
    candidates = []
    for hit in res['hits']['hits']:
        source = hit['_source']
        candidates.append({
            'es_score': hit['_score'],
            **source,
        })
    return candidates


"""Model Preparation"""

model_id = '/nfs/primary/rag_finetuned'
token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_id, token="")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda",
    use_cache=True,
    low_cpu_mem_usage=True,
    token=token
)
model.eval()
model.generation_config.pad_token_id = tokenizer.pad_token_id


def gen_response(messages):
    with torch.no_grad():
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )

        input_ids = tokenizer([input_text], return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True,
                                        skip_special_tokens=True)
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        generate_kwargs = dict(
            input_ids,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            top_k=1,
            temperature=0.2,
            num_beams=1,
            repetition_penalty=1.1,
            eos_token_id=terminators
        )
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        response = ""
        for new_token in streamer:
            # print(new_token, end="")
            response += new_token
            yield new_token
        t.join()


"""Re ranker"""
re_rank_model_id = 'castorini/monot5-base-msmarco-10k'
re_ranker = Reranker(re_rank_model_id, verbose=False, device="cuda")


def re_score(query, candidates):
    docs = [d["segment"] for d in candidates]
    reranked_results = re_ranker.rank(query=query, docs=docs)
    arg_sorted = [r.doc_id for r in reranked_results.top_k(100)]
    reranked_candidates = [candidates[i] for i in arg_sorted]
    for i, cands in enumerate(reranked_candidates):
        cands["monot5_score"] = reranked_results.top_k(100)[i].score
    return reranked_candidates


"""Answer Post process"""


def cite_answer(answer, candidates, threshold):
    sentences = nltk.sent_tokenize(answer)
    cited_answer = []
    for sent in sentences:
        ranked_docs = re_ranker.rank(query=sent, docs=candidates)

        citations = [r.doc_id for r in ranked_docs.top_k(100) if r.score > threshold]
        cited_answer.append({"text": sent, "citations": sorted(citations)})

    return cited_answer


def parse_response(response):
    sentences = re.split("(\[[0-9\[\]]+\])", response)
    answers = []
    for i in range(0, len(sentences) - 1, 2):
        citations = sentences[i + 1].replace("[", " ").replace("]", "").strip().split()
        answers.append({
            "text": sentences[i].strip(),
            "citations": [int(c) for c in citations]
        })

    return answers


with gr.Blocks(title="RAG Demo", css=css_path, theme=gr.themes.Default()) as app:
    def highlight_words(doc_id, chatbot, retrieved_docs):
        if doc_id > len(retrieved_docs):
            return []

        answer = chatbot[-1][-1]
        answer = nltk.word_tokenize(answer)
        answer = [a for a in answer if a not in stop_words]
        doc = retrieved_docs[doc_id - 1]['segment']
        doc = doc.replace("\n", "<newline>")
        result = []

        for w in doc.split():
            temp_w = w.strip(".").strip(",")
            if temp_w in answer:
                result.append((w + " ", "+"))
            else:
                result.append((w.replace("<newline>", "\n") + " ", None))
        return result


    gr.Markdown("# RAG System Demo")
    with gr.Row():
        retrieved_docs = gr.State(value=[])
        with gr.Column(scale=1):
            retriever = gr.Radio(
                ["bm25", "bm25+rerank", "semantic"], label="Retrieval Method"
            )
            num_doc = gr.Slider(0, 20, step=1, label="Number of Documents")
            gr.Markdown("***")
            model_name = gr.Radio(
                ["Llama-3-8B", "Mistral-7B-v3.0"], label="Model"
            )
            gen_trat = gr.Radio(
                ["ICL", "Fine-tuned", "Post Cite"], label="Generate Strategy"
            )
            threshold = gr.Slider(0, 1, label="Cite threshold")
            config_btn = gr.Button("Save")
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            with gr.Row():
                reset_btn = gr.Button(value="Reset", scale=1, size="sm", min_width=10)
                msg = gr.Textbox(show_label=False, autofocus=True, scale=10)
                send_btn = gr.Button(value="Send", scale=1, variant="", size="sm", elem_id="send_btn", min_width=10)
        with gr.Column(scale=2):
            with gr.Tab("Retrieval Docs"):
                # retrieval = gr.Markdown(value=ex, elem_id="debug")
                retrieval_hl = gr.HighlightedText(
                    show_label=False,
                    label="Document",
                    combine_adjacent=True,
                    value=[],
                    color_map={"+": "green"}, show_inline_category=False, elem_id="debug")
                doc_id = gr.Number(show_label=False, visible=False)
                examples = gr.Examples(examples=[[i] for i in range(1, 11)],
                                       inputs=[doc_id, chatbot, retrieved_docs],
                                       outputs=[retrieval_hl],
                                       fn=highlight_words,
                                       run_on_click=True
                                       )
            with gr.Tab("Retrieval Results"):
                retrieval = gr.Json(value={}, elem_id="debug")
            with gr.Tab("Generation Output"):
                answer = gr.Json(value={}, elem_id="debug")


    def on_config_save(model_name, strategy):

        global model, tokenizer, token
        del model
        del tokenizer

        if model_name == "Llama-3-8B":
            if strategy == "Fine-tuned":
                model_id = '/nfs/primary/rag_finetuned'
            else:
                model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        else:
            if strategy == "Fine-tuned":
                model_id = '/nfs/primary/rag_mistral'
            else:
                model_id = "mistralai/Mistral-7B-Instruct-v0.3"

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda",
            use_cache=True,
            low_cpu_mem_usage=True,
            token=token
        )
        model.eval()
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        return ""


    def respond(query, chat_history, retriever, doc_num, model_name, strategy, cite_threshold):
        responses = [
            *chat_history,
            (query, "Retrieving 100 documents ...")
        ]
        yield "", responses, {}, {}, []

        if "sematic" in retriever:
            doc_candidates = index.search(query, top_k=20)
            doc_candidates = [{**c['doc'], "score": c['score']} for c in doc_candidates]
        else:
            doc_candidates = retrieval_candidates(es_conn, query, top_k=100)

        if "rerank" in retriever:
            responses = [
                *chat_history,
                (query, "Re Ranking to get top 20 ...")
            ]
            yield "", responses, doc_candidates[:doc_num], {}, []
            doc_candidates = re_score(query, doc_candidates)

        doc_candidates = doc_candidates[:doc_num]
        responses = [
            *chat_history,
            (query, "Thinking to Answer ...")
        ]
        yield "", responses, doc_candidates, {}, []
        if model_name == "Llama-3-8B":
            if strategy == "Fine-tuned":
                system, _ = get_finetuned_input_llama(query, doc_candidates)
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ]
            elif strategy == "Post Cite":
                system, _ = get_post_cite_input(query, doc_candidates)
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ]
            else:
                system, _ = get_icl_input_llama(query, doc_candidates)
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ]
        else:
            if strategy == "Fine-tuned":
                _, system = get_finetuned_input_mistral(query, doc_candidates)
                messages = [
                    {"role": "user", "content": system},
                ]
            elif strategy == "Post Cite":
                system, _ = get_post_cite_input(query, doc_candidates)
                messages = [
                    {"role": "user", "content": system + f"\n Question: {query}"},
                ]
            else:
                _, system = get_icl_input_mistral(query, doc_candidates)
                messages = [
                    {"role": "user", "content": system},
                ]

        print(messages)
        response = ""
        for token in gen_response(messages):
            response += token
            responses = [
                *chat_history,
                (query, response.replace("assistant\n\n", ""))
            ]
            yield "", responses, doc_candidates, {}, []

        response = response.replace("assistant\n\n", "")

        if strategy == "Post Cite":
            answer_json = cite_answer(response, [d['segment'] for d in doc_candidates], cite_threshold)
        else:
            answer_json = parse_response(response)
        yield "", responses, doc_candidates, answer_json, doc_candidates


    send_btn.click(respond, [msg, chatbot, retriever, num_doc, model_name, gen_trat, threshold],
                   [msg, chatbot, retrieval, answer, retrieved_docs])
    msg.submit(respond, [msg, chatbot, retriever, num_doc, model_name, gen_trat, threshold],
               [msg, chatbot, retrieval, answer, retrieved_docs])
    config_btn.click(on_config_save, [model_name, gen_trat],
                     [msg])
    # examples.submit(highlight_words, [doc_id, chatbot, retrieved_docs], [retrieval_hl])

if __name__ == "__main__":
    app.launch(
        share=True,
        # server_port=1111
    )