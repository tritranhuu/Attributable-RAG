import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import T5ForConditionalGeneration, T5Tokenizer

from threading import Thread
from .prompt_builder import *
import json

class Generator:
    def __init__(self, model_id, arch, device=None, hf_token=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="cuda",
                use_cache=True,
                low_cpu_mem_usage=True,
                token=hf_token
        )
        self.prompt_builder = PROMPT_BUILDER_DICT[arch]

    def gen_response(self, candidates, query):
        messages = self.prompt_builder(candidates, query)
        with torch.no_grad():
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False
            )

            input_ids = self.tokenizer([input_text], return_tensors="pt").to("cuda")

            streamer = TextIteratorStreamer(self.tokenizer, timeout=10., skip_prompt=True,
                                                            skip_special_tokens=True)
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
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
            t = Thread(target=self.model.generate, kwargs=generate_kwargs)
            t.start()
            response = ""
            for new_token in streamer:
                response += new_token
                # yield new_token
            t.join()
            return response.replace("assistant\n\n", "").strip()