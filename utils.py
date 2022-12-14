import itertools
import json
import linecache
import os
import pickle
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import git
import numpy as np
import torch
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from torch import nn
from torch.utils.data import Dataset, Sampler

from transformers import BartTokenizer


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def encode_line(tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
    """Only used by LegacyDataset"""
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
    result = tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        **extra_kw,
    )
    result["text"] = line
    return result

BART_TOKENIZER_STARTCHAR='Ġ'
# encode the relevance file line: the format is the same as the source file, each word has an answer relevance probability
def encode_relevance_line(tokenizer, src_line, relevance_line, pad_token_id, max_length, baseline, pad_to_max_length=True, return_tensors="pt"):
    if baseline:
        src_context_line = src_line.split('[SEP]')[0]

        src_tokens = tokenizer.tokenize(src_context_line)  # since extra_kw = {"add_prefix_space": True}
        src_tokens = src_tokens[:-1]
        src_line_tokens = tokenizer.tokenize(src_line)

        relevance_2_ids = []
        relevance_2_ids.append(0.0)
        cur_word_index = 0

        for index, token in enumerate(src_tokens):
            relevance_2_ids.append(0.0)

        for i in src_line_tokens[index + 1: index + 5]:  # [SEP]
            relevance_2_ids.append(0.0)

        for i in src_line_tokens[index + 5:]:
            relevance_2_ids.append(0.0)

        relevance_2_ids.append(0.0)

        # padding
        while len(relevance_2_ids) < max_length:
            relevance_2_ids.append(float(pad_token_id))
        # truncate
        if len(relevance_2_ids) > max_length:
            relevance_2_ids = relevance_2_ids[:max_length]

        return torch.FloatTensor(relevance_2_ids)

    else:

        src_context_line = src_line.split('[SEP]')[0]
        assert len(src_context_line.split()) == len(relevance_line.split()), f'the length is {len(src_context_line.split())} and {len(relevance_line.split())}'

        relevance_2_words = []
        relevance_2_words = relevance_line.split()
        for i, item in enumerate(relevance_2_words):
            relevance_2_words[i] = float(item)
        src_tokens = tokenizer.tokenize(src_context_line) # since extra_kw = {"add_prefix_space": True}
        src_tokens = src_tokens[:-1]
        src_line_tokens = tokenizer.tokenize(src_line)

        relevance_2_ids = []
        relevance_2_ids.append(0.0)
        cur_word_index = 0

        for index, token in enumerate(src_tokens):
            if index==0:
                cur_word_index = 0
            else:
                if token.startswith(BART_TOKENIZER_STARTCHAR): # a new word
                    # relevance_2_ids.append(float(relevance_2_words[cur_word_index]))
                    if token.replace(BART_TOKENIZER_STARTCHAR, '') != '':
                        cur_word_index += 1
            relevance_2_ids.append(float(relevance_2_words[cur_word_index]))

        que_atten = max(relevance_2_words)


        for i in src_line_tokens[index + 1 : index + 5]: #[SEP]
            relevance_2_ids.append(0.0)

        for i in src_line_tokens[index + 5 : ]:
            relevance_2_ids.append(float(que_atten))

        relevance_2_ids.append(0.0)

        # padding
        while len(relevance_2_ids) < max_length:
            relevance_2_ids.append(float(pad_token_id))
        # truncate
        if len(relevance_2_ids) > max_length:
            relevance_2_ids = relevance_2_ids[:max_length]

        return torch.FloatTensor(relevance_2_ids)


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": round(corpus_bleu(output_lns, [refs_lns], **kwargs).score, 4)}


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


class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.add_prefix_space = isinstance(self.tokenizer, BartTokenizer)

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def make_sortish_sampler(self, batch_size):
        return SortishSampler(self.src_lens, batch_size)

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


class LegacySeq2SeqDataset(AbstractSeq2SeqDataset):
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Call tokenizer on src and tgt_lines"""
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        source_inputs = encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": y,
        }
        return batch

class Seq2SeqDataset_QFS(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
        baseline = False,
        raw= False,
        weos=False
    ):
        super().__init__()
        if not raw:
            self.src_file = Path(data_dir).joinpath(type_path + "_content")
            self.tgt_file = Path(data_dir).joinpath(type_path + "_summary")
            #todo: change the score to 0 for baseline
            self.relevance_file = Path(data_dir).joinpath(type_path + "_relevance")   # the qa relevance score
            self.src_lens = self.get_char_lens(self.src_file)
            self.rel_lens = self.get_char_lens(self.relevance_file)
            self.max_source_length = max_source_length
            self.max_target_length = max_target_length
            assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
            assert min(self.rel_lens) > 0, f"found empty line in {self.relevance_file}"

            self.tokenizer = tokenizer
            self.prefix = prefix #todo: why a  prefix?
            if n_obs is not None:
                self.src_lens = self.src_lens[:n_obs]
            self.pad_token_id = self.tokenizer.pad_token_id
            self.src_lang = src_lang
            self.tgt_lang = tgt_lang
            self.baseline = baseline
            self.raw = raw
        else:
            self.src_file = Path(data_dir).joinpath(type_path + "_content")
            self.query_file = Path(data_dir).joinpath(type_path + "_query")
            self.tgt_file = Path(data_dir).joinpath(type_path + "_summary")
            self.relevance_file = Path(data_dir).joinpath(type_path + ".relevance")  # the qa relevance score
            self.src_lens = self.get_char_lens(self.src_file)
            # self.rel_lens = self.get_char_lens(self.relevance_file)
            self.max_source_length = max_source_length
            self.max_target_length = max_target_length
            assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
            # assert min(self.rel_lens) > 0, f"found empty line in {self.relevance_file}"

            self.tokenizer = tokenizer
            self.prefix = prefix  # todo: why a  prefix?
            if n_obs is not None:
                self.src_lens = self.src_lens[:n_obs]
            self.pad_token_id = self.tokenizer.pad_token_id
            self.src_lang = src_lang
            self.tgt_lang = tgt_lang
            self.baseline = baseline
            self.raw = raw
            self.weos=weos

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        # print(self.raw)
        if not self.raw:
            index = index + 1  # linecache starts at 1
            source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n") # 为什么加了一个prefix呢？
            tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
            relevance_line = linecache.getline(str(self.relevance_file), index).rstrip("\n")
            assert source_line, f"empty source line for index {index}"
            assert tgt_line, f"empty tgt line for index {index}"
            assert relevance_line, f"empty relevance line for index {index}"

            source_inputs = encode_line(self.tokenizer, source_line, self.max_source_length)
            target_inputs = encode_line(self.tokenizer, tgt_line, self.max_target_length)

            pad_token_id = self.pad_token_id
            relevance_inputs = encode_relevance_line(self.tokenizer, source_line, relevance_line, pad_token_id, self.max_source_length,self.baseline)


            source_ids = source_inputs["input_ids"].squeeze()
            target_ids = target_inputs["input_ids"].squeeze()
            src_mask = source_inputs["attention_mask"].squeeze()
            relevance_atten = relevance_inputs.squeeze()
            assert source_ids.size() == relevance_atten.size(), f'the size of the source_id and relevance_inputs is {source_ids.size()} and {relevance_atten.size()}'

            return {
                "input_ids": source_ids,
                "attention_mask": src_mask,
                "decoder_input_ids": target_ids,
                "answer_relevance_atten": relevance_atten,
                "text": source_inputs["text"],
            }
        else:
            index = index + 1  # linecache starts at 1
            source_line = self.prefix + linecache.getline(str(self.src_file), index).replace('<s>','').replace('<eos>', '</s>').rstrip("\n").strip() # 为什么加了一个prefix呢？
            query_line = linecache.getline(str(self.query_file), index).rstrip("\n").replace('<s>','').replace('<eos>', '').strip()
            source_line = ' '.join([source_line, query_line])
            if not self.weos:
                tgt_line = linecache.getline(str(self.tgt_file), index).replace('<s>','').replace('<eos>', '').rstrip("\n").strip() # todo: whether to change?
            else:
                tgt_line = linecache.getline(str(self.tgt_file), index).replace('<eos>','').replace('<s>','').rstrip("\n").strip() # todo: whether to change?
            # relevance_line = linecache.getline(str(self.relevance_file), index).rstrip("\n")
            # print(source_line, tgt_line)
            assert source_line, f"empty source line for index {index}"
            assert tgt_line, f"empty tgt line for index {index}"
            # assert relevance_line, f"empty relevance line for index {index}"
            relevance_line = None
            source_inputs = encode_line(self.tokenizer, source_line, self.max_source_length)
            target_inputs = encode_line(self.tokenizer, tgt_line, self.max_target_length)

            pad_token_id = self.pad_token_id
            relevance_inputs = encode_relevance_line(self.tokenizer, source_line, relevance_line, pad_token_id,
                                                     self.max_source_length, self.baseline)

            source_ids = source_inputs["input_ids"].squeeze()
            target_ids = target_inputs["input_ids"].squeeze()
            src_mask = source_inputs["attention_mask"].squeeze()
            relevance_atten = relevance_inputs.squeeze()
            assert source_ids.size() == relevance_atten.size(), f'the size of the source_id and relevance_inputs is {source_ids.size()} and {relevance_atten.size()}'
            if index %1000 == 0:
                pass
                # print(source_line, tgt_line,source_inputs, target_inputs)
            return {
                "input_ids": source_ids,
                "attention_mask": src_mask,
                "decoder_input_ids": target_ids,
                "answer_relevance_atten": relevance_atten,
                "text": source_inputs["text"],
            }
    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        ans_relevance_atten = torch.stack([x["answer_relevance_atten"] for x in batch])

        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        source_ans_relevance_atten, _ = trim_batch(ans_relevance_atten, float(pad_token_id), attention_mask=masks)
        source_ans_relevance_atten = source_ans_relevance_atten.to(torch.float16)

        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": y,
            # "decoder_input_ids": y,
            "answer_relevance_atten": source_ans_relevance_atten,
        }
        return batch

    def make_sortish_sampler(self, batch_size):
        return SortishSampler(self.src_lens, batch_size)


class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {
            "tgt_texts": tgt_line,
            "src_texts": source_line,
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            src_lang=self.src_lang,
            tgt_texts=[x["tgt_texts"] for x in batch],
            tgt_lang=self.tgt_lang,
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            add_prefix_space=self.add_prefix_space,
        )
        return batch_encoding.data


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size):
        self.data, self.bs = data, batch_size

    def key(self, i):
        return self.data[i]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data))
        sz = self.bs * 50
        ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)


logger = getLogger(__name__)


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_git_info(folder_path: str) -> None:
    """Save git information to output_dir/git_log.json"""
    repo_infos = get_git_info()
    save_json(repo_infos, os.path.join(folder_path, "git_log.json"))


def save_json(content, path):
    with open(path, "w") as f:
        json.dump(content, f, indent=4)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
    }
    return repo_infos


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def calculate_rouge(output_lns: List[str], reference_lns: List[str], use_stemmer=True) -> Dict:
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}


# Utilities for freezing parameters and checking whether they are frozen


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"
