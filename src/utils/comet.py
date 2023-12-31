import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List
from more_itertools import chunked

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        model.config.update(pars)


class Comet:
    def __init__(self, model_path, device):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.batch_size = 1

        # init params
        use_task_specific_params(self.model, "summarization")
        self.model.zero_grad()

    def generate(self, input_event, rel):
        query = "{} {} [GEN]".format(input_event, rel)

        with torch.no_grad():
            query = self.tokenizer(
                query, return_tensors="pt", truncation=True, padding="max_length"
            ).to(self.device)
            input_ids, attention_mask = trim_batch(
                **query, pad_token_id=self.tokenizer.pad_token_id
            )

            summaries = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_start_token_id=None,
                num_beams=5,
                num_return_sequences=5,
            )

            dec = self.tokenizer.batch_decode(
                summaries,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return dec


class CometMulti:
    def __init__(self, model_path, device, batch_size=1):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.batch_size = batch_size  # 设置批处理大小

        # 初始化其他参数
        use_task_specific_params(self.model, "summarization")
        self.model.zero_grad()

    def generate(self, input_events: List[str], rel)->List[List[str]]:
        # 每次推理都根据批处理大小返回结果
        
        # if len(input_events) != self.batch_size:
        #     return

        queries = ["{} {} [GEN]".format(event, rel) for event in input_events]

        with torch.no_grad():
            queries = self.tokenizer(
                queries, return_tensors="pt", truncation=True, padding="max_length"
            ).to(self.device)
            input_ids, attention_mask = trim_batch(
                **queries, pad_token_id=self.tokenizer.pad_token_id
            )

            summaries = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_start_token_id=None,
                num_beams=5,
                num_return_sequences=5,
            )

            dec = self.tokenizer.batch_decode(
                summaries,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        batch_results=list(chunked(dec,5))

        return batch_results
