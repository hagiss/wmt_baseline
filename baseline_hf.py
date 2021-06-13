import numpy as np
import os
import sys
import logging
import inspect

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import argparse
from pathlib import Path
from datasets import load_dataset, load_metric
import datasets
import random
import pandas as pd
import torch.nn.functional as F
import torch

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    MarianMTModel,
    MarianModel,
)
from transformers import MarianConfig, MarianMTModel

from dataset import TranslationDataset

import ipdb

logger = logging.getLogger(__name__)

"""
Add arguments here
"""


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})

    data_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on "
                    "a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacreblue) on " "a jsonlines file."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
                    "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
                    "needs to be the target language token.(Usually it is the target language token)"
        },
    )

    def __post_init__(self):
        """
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        elif self.source_lang is None or self.target_lang is None:
            raise ValueError("Need to specify the source language and the target language.")

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension == "json", "`train_file` should be a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension == "json", "`validation_file` should be a json file."
        """
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
        print(self.val_max_target_length, self.max_target_length)


criterion = torch.nn.CrossEntropyLoss()

class new_Seq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        inputs['return_dict'] = True
        inputs['output_hidden_states'] = True
        inputs['output_attentions'] = True
        outputs = model(**inputs)
        batch, seq, di = outputs.encoder_hidden_states[0].shape

        encoder_mask = outputs.encoder_attentions[0].detach()[:, 0, 0, :].bool().unsqueeze(-1).expand([batch, seq, di]).float()
        decoder_mask = inputs['labels'] >= 0
        _, d_seq = decoder_mask.shape
        decoder_mask = decoder_mask.unsqueeze(-1).expand([batch, d_seq, di]).float()

        info_nce_labels = torch.arange(batch).cuda()

        encoder_embed = (encoder_mask*outputs.encoder_hidden_states[0]).sum(dim=1) / (encoder_mask!=0).sum(dim=1)
        decoder_embed = (decoder_mask*outputs.decoder_hidden_states[0]).sum(dim=1) / (decoder_mask!=0).sum(dim=1)

        encoder_embed = model.mlp1(encoder_embed)
        decoder_embed = model.mlp1(decoder_embed)

        encoder_embed = F.normalize(encoder_embed.detach(), dim=1)
        decoder_embed = F.normalize(decoder_embed, dim=1)

        similarity_matrix = torch.matmul(encoder_embed, decoder_embed.T) / 0.1

        # sim_matrix = torch.matmul(encoder_embed, encoder_embed.T) / 0.1
        # mask = torch.ones_like(sim_matrix).scatter_(1, info_nce_labels.unsqueeze(1), 0.).cuda()
        # similarity_matrix = torch.cat([similarity_matrix, sim_matrix[mask.bool()].view(batch, -1)], dim=1)

        info_nce_loss = criterion(similarity_matrix, info_nce_labels)

        # print(encoder_embed.shape)
        # print(decoder_embed.shape)
        # print(inputs)
        # outputs = outputs.loss
        # print(outputs.decoder_hidden_states.shape)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        loss += 0.5 * info_nce_loss

        return (loss, outputs) if return_outputs else loss


class MLP(torch.nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    train_data = [line.rstrip('\n').split('\t') for line in
                  open(Path(data_args.data_path) / 'train.tsv', 'r').readlines()]
    dev_data = [line.rstrip('\n').split('\t') for line in open(Path(data_args.data_path) / 'dev.tsv', 'r').readlines()]
    test_data = [line.rstrip('\n').split('\t') for line in
                 open(Path(data_args.data_path) / 'test.tsv', 'r').readlines()]

    if data_args.max_train_samples is not None:
        # Recommended to use a pre-shuffled train set
        train_data = train_data[:data_args.max_train_samples]

    """
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)


    if args.from_scratch:
        config = MarianConfig.from_pretrained(model_args.model_name_or_path)
        model = MarianMTModel(config)
        #ipdb.set_trace()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    """
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,

    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,

    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,

    )

    model.mlp1 = MLP(512, 256)

    # print(inspect.getsource(model.forward))
    model.resize_token_embeddings(len(tokenizer))
    # model.forward = alternative_forward.__get__(model, MarianMTModel)
    # model.model.forward = al_forward2.__get__(model.model, MarianModel)

    # Metric for evaluating generated translations
    metric = load_metric("sacrebleu")
    # metric = Sacrebleu

    """
    batch_size = 32
    args = Seq2SeqTrainingArguments(
            "data_2_scratch",
            warmup_steps=0,
            evaluation_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=3,
            predict_with_generate=True,
            fp16=True,
    )
    """

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    max_input_length = data_args.max_source_length
    max_target_length = data_args.max_target_length
    source_lang = "en"
    target_lang = "fr"

    train_dataset = TranslationDataset(train_data, tokenizer, max_input_length, max_target_length, source_lang,
                                       target_lang)
    eval_dataset = TranslationDataset(dev_data, tokenizer, max_input_length, max_target_length, source_lang,
                                      target_lang)
    test_dataset = TranslationDataset(test_data, tokenizer, max_input_length, max_target_length, source_lang,
                                      target_lang)

    # dd = train_dataset[0]

    trainer = new_Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,

    )

    # ipdb.set_trace()

    """
    trainer = Seq2SeqTrainer(
                model,
                args,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
     )
    """
    # print(dir(trainer))
    # """
    # Train the model
    if training_args.do_train:
        checkpoint = None
        # if training_args.resume_from_checkpoint is not None:
        #    checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #    checkpoint = last_checkpoint
        # train_result = trainer.train(resume_from_checkpoint=checkpoint)
        print('Starting training')
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate on the Dev Set
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Evaluate on the Test Set
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset,
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # ipdb.set_trace()

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    # if training_args.push_to_hub:
    #    trainer.push_to_hub()

    return results
    # """


if __name__ == "__main__":
    main()
