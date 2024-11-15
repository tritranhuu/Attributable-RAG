from elasticsearch import Elasticsearch, helpers, RequestsHttpConnection
from rerankers import Reranker

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
import json
from tqdm import tqdm

# re_rank_model_id = 'castorini/monobert-large-msmarco-finetune-only'
# re_rank_model_id = 'castorini/monot5-large-msmarco-10k'
re_rank_model_id = 'mixedbread-ai/mxbai-rerank-large-v1'
# re_rank_model_id = 'castorini/rankllama-v1-7b-lora-passage'
# re_rank_model_id = "zephyr"
# re_rank_model_id = "castorini/tct_colbert-v2-hnp-msmarco"
re_ranker = Reranker(re_rank_model_id, verbose=False, model_type = "cross-encoder")


def re_score(query, candidates):
    docs = [f"{d['title']}\n{d['headings']}\n{d['segment']}" for d in candidates]
    reranked_results = re_ranker.rank(query=query, docs=docs)
    arg_sorted = [r.doc_id for r in reranked_results.top_k(1000)]
    reranked_candidates = [candidates[i] for i in arg_sorted]
    for i, cands in enumerate(reranked_candidates):
        cands["rerank_score"] = reranked_results.top_k(1000)[i].score
    return reranked_candidates

## Rerank Llama
def get_model(peft_model_name, token):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=1, token=token,
        torch_dtype=torch.float16,
        device_map="cuda")
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

# Load the tokenizer and model
token = ""
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=token)
rank_llama_model = get_model('castorini/rankllama-v1-7b-lora-passage', token)

def re_score_llamaRank(query, candidates):
    for cand in candidates:
        try:
            doc = f"{cand['title']}\n{cand['headings']}\n{cand['segment']}"
            inputs = tokenizer(f'query: {query}', f'document: {doc}', return_tensors='pt').to("cuda")
            with torch.no_grad():
                outputs = rank_llama_model(**inputs)
                logits = outputs.logits
                score = float(logits[0][0].detach().cpu().numpy())
                cand["rr_score"] = score
        except:
            doc = f"{cand['title']}\n{cand['segment']}"
            inputs = tokenizer(f'query: {query}', f'document: {doc}', return_tensors='pt').to("cuda")
            with torch.no_grad():
                outputs = rank_llama_model(**inputs)
                logits = outputs.logits
                score = float(logits[0][0].detach().cpu().numpy())
                cand["rr_score"] = score
    reranked_candidates = sorted(candidates, key=lambda x: x["rr_score"], reverse=True)
    return reranked_candidates