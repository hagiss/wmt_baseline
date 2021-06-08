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
import sacrebleu as scb

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


def al_forward2(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):
    r"""
    Returns:

    Example::

        >>> from transformers import MarianTokenizer, MarianModel

        >>> tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        >>> model = MarianModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')

        >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("<pad> Studien haben gezeigt dass es hilfreich ist einen Hund zu besitzen",
        ... return_tensors="pt", add_special_tokens=False).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

        >>> last_hidden_states = outputs.last_hidden_state
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if encoder_outputs is None:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )
    print(encoder_outputs.hidden_states)

    # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_outputs[0],
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        encoder_head_mask=head_mask,
        past_key_values=past_key_values,
        inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    if not return_dict:
        return decoder_outputs + encoder_outputs

    return {
        'last_hidden_state': decoder_outputs.last_hidden_state,
    }

    # return Seq2SeqModelOutput(
    #     last_hidden_state=decoder_outputs.last_hidden_state,
    #     past_key_values=decoder_outputs.past_key_values,
    #     decoder_hidden_states=decoder_outputs.hidden_states,
    #     decoder_attentions=decoder_outputs.attentions,
    #     cross_attentions=decoder_outputs.cross_attentions,
    #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
    #     encoder_hidden_states=encoder_outputs.hidden_states,
    #     encoder_attentions=encoder_outputs.attentions,
    # )


def alternative_forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
        Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
        config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
        (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

    Returns:

    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if labels is not None:
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

    outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs,
        decoder_attention_mask=decoder_attention_mask,
        head_mask=head_mask,
        decoder_head_mask=decoder_head_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        decoder_inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    # print(inspect.getsource(self.model.forward))
    print(outputs.encoder_hidden_states[0].shape)
    print(outputs.decoder_hidden_states[0].shape)
    lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
    print(labels.shape)

    masked_lm_loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

    if not return_dict:
        output = (lm_logits,) + outputs[1:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    return Seq2SeqLMOutput(
        loss=masked_lm_loss,
        logits=lm_logits,
        past_key_values=outputs.past_key_values,
        decoder_hidden_states=outputs.decoder_hidden_states,
        decoder_attentions=outputs.decoder_attentions,
        cross_attentions=outputs.cross_attentions,
        encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        encoder_hidden_states=outputs.encoder_hidden_states,
        encoder_attentions=outputs.encoder_attentions,
    )


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

        # print((encoder_mask*outputs.encoder_hidden_states[0]).sum(dim=1).shape)
        # print((encoder_mask!=0).sum(dim=1).shape)

        encoder_embed = (encoder_mask*outputs.encoder_hidden_states[0]).sum(dim=1) / (encoder_mask!=0).sum(dim=1)
        decoder_embed = (decoder_mask*outputs.decoder_hidden_states[0]).sum(dim=1) / (decoder_mask!=0).sum(dim=1)

        encoder_embed = F.normalize(encoder_embed, dim=1)
        decoder_embed = F.normalize(decoder_embed, dim=1)


        info_nce_labels = torch.arange(batch).cuda()
        # print(info_nce_labels)

        similarity_matrix = torch.matmul(encoder_embed, decoder_embed.T) / 0.1

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

        loss += info_nce_loss

        return (loss, outputs) if return_outputs else loss

class Sacrebleu(datasets.Metric):
    def _info(self):
        if version.parse(scb.__version__) < version.parse("1.4.12"):
            raise ImportWarning(
                "To use `sacrebleu`, the module `sacrebleu>=1.4.12` is required, and the current version of `sacrebleu` doesn't match this condition.\n"
                'You can install it with `pip install "sacrebleu>=1.4.12"`.'
            )
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/mjpost/sacreBLEU",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                }
            ),
            codebase_urls=["https://github.com/mjpost/sacreBLEU"],
            reference_urls=[
                "https://github.com/mjpost/sacreBLEU",
                "https://en.wikipedia.org/wiki/BLEU",
                "https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213",
            ],
        )

    def _compute(
        self,
        predictions,
        references,
        smooth_method="exp",
        smooth_value=None,
        force=False,
        lowercase=False,
        tokenize=scb.DEFAULT_TOKENIZER,
        use_effective_order=False,
    ):
        references_per_prediction = len(references[0])
        if any(len(refs) != references_per_prediction for refs in references):
            raise ValueError("Sacrebleu requires the same number of references for each prediction")
        transformed_references = [[refs[i] for refs in references] for i in range(references_per_prediction)]
        output = scb.corpus_bleu(
            sys_stream=predictions,
            ref_streams=transformed_references,
            smooth_method=smooth_method,
            smooth_value=smooth_value,
            force=force,
            lowercase=lowercase,
            tokenize=tokenize,
            use_effective_order=use_effective_order,
        )
        output_dict = {
            "score": output.score,
            "counts": output.counts,
            "totals": output.totals,
            "precisions": output.precisions,
            "bp": output.bp,
            "sys_len": output.sys_len,
            "ref_len": output.ref_len,
        }
        return output_dict


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

    # print(inspect.getsource(model.forward))
    model.resize_token_embeddings(len(tokenizer))
    # model.forward = alternative_forward.__get__(model, MarianMTModel)
    # model.model.forward = al_forward2.__get__(model.model, MarianModel)

    # Metric for evaluating generated translations
    # metric = load_metric("sacrebleu")
    metric = Sacrebleu

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
