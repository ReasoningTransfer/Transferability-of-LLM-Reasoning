# """
# Know What You Don’t Know: Unanswerable Questions for SQuAD
# https://arxiv.org/pdf/1806.03822.pdf

# Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset,
# consisting of questions posed by crowdworkers on a set of Wikipedia articles,
# where the answer to every question is a segment of text, or span, from the
# corresponding reading passage, or the question might be unanswerable.
# SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable
# questions written adversarially by crowdworkers to look similar to answerable ones.
# To do well on SQuAD2.0, systems must not only answer questions when possible, but
# also determine when no answer is supported by the paragraph and abstain from answering.

# Homepage: https://rajpurkar.github.io/SQuAD-explorer/
# """
from functools import partial
from math import exp
import re

import datasets
from packaging import version

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask

_CITATION = """
@misc{rajpurkar2018know,
    title={Know What You Don't Know: Unanswerable Questions for SQuAD},
    author={Pranav Rajpurkar and Robin Jia and Percy Liang},
    year={2018},
    eprint={1806.03822},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


def _squad_metric(predictions, references):
    import evaluate

    squad_metric = evaluate.load("squad_v2")
    return squad_metric.compute(predictions=predictions, references=references)


def _squad_agg(key, items):
    predictions, references = zip(*items)
    return _squad_metric(predictions=predictions, references=references).get(key, 0)


# Helper functions for parsing boxed answers

def strip_string(string: str) -> str:
    s = string.replace("\n", "").replace("\\", "\\").replace(" ", "")
    return s


def remove_boxed(s: str) -> str:
    if s.startswith("\\fbox{"):
        return s[len("\\fbox{"):-1]
    if s.startswith("\\boxed "):
        return s[len("\\boxed "):]
    if s.startswith("\\boxed{"):
        return s[len("\\boxed{"):-1]
    return s


def last_boxed_only_string(string: str) -> str:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    depth = 0
    for i in range(idx, len(string)):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                return string[idx : i + 1]
    return None


def is_equiv(str1: str, str2: str) -> bool:
    if str1 is None or str2 is None:
        return False
    return strip_string(str1) == strip_string(str2)


class SQuAD2(ConfigurableTask):
    VERSION = 3
    DATASET_PATH = "squad_v2"
    DATASET_NAME = None

    def __init__(self, config=None):
        super().__init__(config={"metadata": {"version": self.VERSION}})

    # Ensure correct datasets version
    assert version.parse(datasets.__version__) >= version.parse("1.11.0"), (
        "datasets v1.11.0 or later required for SQuAD"
    )

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return (
            "Title: " + doc["title"] + "\n\n"
            "Background: " + doc["context"] + "\n\n"
            "Question: " + doc["question"] + "\n\n"
            # Prompt to place final answer in \boxed{}
            "Answer: Please put the final answer in the \\boxed{}.\n"
        )

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        answers = doc["answers"]["text"]
        if answers:
            return " " + answers[0]
        else:
            return " unanswerable"

    def construct_requests(self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs):
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, {"until": ["\n"]}),
                idx=0,
                **kwargs,
            ),
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, " unanswerable"),
                idx=0,
                **kwargs,
            ),
        ]

    def process_results(self, doc, results):
        continuation, (logprob_unanswerable, _) = results

        # Extract boxed answer if present
        boxed = last_boxed_only_string(continuation)
        if boxed:
            answer_text = remove_boxed(boxed).strip()
        else:
            answer_text = continuation.strip()

        no_answer_probability = exp(logprob_unanswerable)

        predictions = {
            "id": doc["id"],
            "prediction_text": answer_text,
            "no_answer_probability": no_answer_probability,
        }

        references = {
            "id": doc["id"],
            "answers": doc["answers"],
        }

        return {
            "exact": (predictions, references),
            "f1": (predictions, references),
            "HasAns_exact": (predictions, references),
            "HasAns_f1": (predictions, references),
            "NoAns_exact": (predictions, references),
            "NoAns_f1": (predictions, references),
            "best_exact": (predictions, references),
            "best_f1": (predictions, references),
        }

    def aggregation(self):
        return {
            "exact": partial(_squad_agg, "exact"),
            "f1": partial(_squad_agg, "f1"),
            "HasAns_exact": partial(_squad_agg, "HasAns_exact"),
            "HasAns_f1": partial(_squad_agg, "HasAns_f1"),
            "NoAns_exact": partial(_squad_agg, "NoAns_exact"),
            "NoAns_f1": partial(_squad_agg, "NoAns_f1"),
            "best_exact": partial(_squad_agg, "best_exact"),
            "best_f1": partial(_squad_agg, "best_f1"),
        }

    def higher_is_better(self):
        return {key: True for key in self.aggregation().keys()}

# from functools import partial
# from math import exp

# import datasets
# from packaging import version

# from lm_eval.api.instance import Instance
# from lm_eval.api.task import ConfigurableTask


# _CITATION = """
# @misc{rajpurkar2018know,
#     title={Know What You Don't Know: Unanswerable Questions for SQuAD},
#     author={Pranav Rajpurkar and Robin Jia and Percy Liang},
#     year={2018},
#     eprint={1806.03822},
#     archivePrefix={arXiv},
#     primaryClass={cs.CL}
# }
# """


# def _squad_metric(predictions, references):
#     import evaluate

#     squad_metric = evaluate.load("squad_v2")
#     return squad_metric.compute(predictions=predictions, references=references)


# def _squad_agg(key, items):
#     predictions, references = zip(*items)

#     return _squad_metric(predictions=predictions, references=references).get(key, 0)


# class SQuAD2(ConfigurableTask):
#     VERSION = 3
#     DATASET_PATH = "squad_v2"
#     DATASET_NAME = None

#     def __init__(self, config=None):
#         super().__init__(config={"metadata": {"version": self.VERSION}})

#     # HF changed squad on us so we have to make sure we aren't running the old one
#     assert version.parse(datasets.__version__) >= version.parse("1.11.0"), (
#         "datasets v1.11.0 or later required for SQuAD"
#     )

#     def has_training_docs(self):
#         return True

#     def has_validation_docs(self):
#         return True

#     def has_test_docs(self):
#         return False

#     def training_docs(self):
#         return self.dataset["train"]

#     def validation_docs(self):
#         return self.dataset["validation"]

#     def doc_to_text(self, doc):
#         return (
#             "Title: "
#             + doc["title"]
#             + "\n\n"
#             + "Background: "
#             + doc["context"]
#             + "\n\n"
#             + "Question: "
#             + doc["question"]
#             + "\n\n"
#             + "Answer:"
#         )

#     def should_decontaminate(self):
#         return True

#     def doc_to_decontamination_query(self, doc):
#         return doc["context"]

#     def doc_to_target(self, doc):
#         answer_list = doc["answers"]["text"]
#         if len(answer_list) > 0:
#             answer = answer_list[0]
#         else:
#             answer = "unanswerable"
#         return " " + answer

#     def construct_requests(
#         self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs
#     ):
#         """Uses RequestFactory to construct Requests and returns an iterable of
#         Requests which will be sent to the LM.

#         :param doc:
#             The document as returned from training_docs, validation_docs, or test_docs.
#         :param ctx: str
#             The context string, generated by fewshot_context. This includes the natural
#             language description, as well as the few shot examples, and the question
#             part of the document for `doc`.
#         """

#         return [
#             Instance(
#                 request_type="generate_until",
#                 doc=doc,
#                 arguments=(ctx, {"until": ["\n"]}),
#                 idx=0,
#                 **kwargs,
#             ),
#             Instance(
#                 request_type="loglikelihood",
#                 doc=doc,
#                 arguments=(ctx, " " + "unanswerable"),
#                 idx=0,
#                 **kwargs,
#             ),
#         ]

#     def process_results(self, doc, results):
#         """Take a single document and the LM results and evaluates, returning a
#         dict where keys are the names of submetrics and values are the values of
#         the metric for that one document

#         :param doc:
#             The document as returned from training_docs, validation_docs, or test_docs.
#         :param results:
#             The results of the requests created in construct_requests.
#         """

#         continuation, (logprob_unanswerable, _) = results

#         no_answer_probability = exp(logprob_unanswerable)

#         predictions = {
#             "id": doc["id"],
#             "prediction_text": continuation,
#             "no_answer_probability": no_answer_probability,
#         }

#         references = {
#             "id": doc["id"],
#             "answers": doc["answers"],
#         }

#         return {
#             "exact": (
#                 predictions,
#                 references,
#             ),  # Exact match (the normalized answer exactly match the gold answer)
#             "f1": (
#                 predictions,
#                 references,
#             ),  # The F-score of predicted tokens versus the gold answer
#             "HasAns_exact": (
#                 predictions,
#                 references,
#             ),  # Exact match (the normalized answer exactly match the gold answer)
#             "HasAns_f1": (
#                 predictions,
#                 references,
#             ),  # The F-score of predicted tokens versus the gold answer
#             "NoAns_exact": (
#                 predictions,
#                 references,
#             ),  # Exact match (the normalized answer exactly match the gold answer)
#             "NoAns_f1": (
#                 predictions,
#                 references,
#             ),  # The F-score of predicted tokens versus the gold answer
#             "best_exact": (
#                 predictions,
#                 references,
#             ),  # Best exact match (with varying threshold)
#             "best_f1": (predictions, references),  # Best F1 (with varying threshold)
#         }

#     def aggregation(self):
#         """
#         :returns: {str: [float] -> float}
#             A dictionary where keys are the names of submetrics and values are
#             functions that aggregate a list of metrics
#         """
#         return {
#             "exact": partial(
#                 _squad_agg, "exact"
#             ),  # Exact match (the normalized answer exactly match the gold answer)
#             "f1": partial(
#                 _squad_agg, "f1"
#             ),  # The F-score of predicted tokens versus the gold answer
#             "HasAns_exact": partial(
#                 _squad_agg, "HasAns_exact"
#             ),  # Exact match (the normalized answer exactly match the gold answer)
#             "HasAns_f1": partial(
#                 _squad_agg, "HasAns_f1"
#             ),  # The F-score of predicted tokens versus the gold answer
#             "NoAns_exact": partial(
#                 _squad_agg, "NoAns_exact"
#             ),  # Exact match (the normalized answer exactly match the gold answer)
#             "NoAns_f1": partial(
#                 _squad_agg, "NoAns_f1"
#             ),  # The F-score of predicted tokens versus the gold answer
#             "best_exact": partial(
#                 _squad_agg, "best_exact"
#             ),  # Best exact match (with varying threshold)
#             "best_f1": partial(
#                 _squad_agg, "best_f1"
#             ),  # Best F1 (with varying threshold)
#         }

#     def higher_is_better(self):
#         """
#         :returns: {str: bool}
#             A dictionary where keys are the names of submetrics and values are
#             whether a higher value of the submetric is better
#         """
#         return {
#             "exact": True,  # Exact match (the normalized answer exactly match the gold answer)
#             "f1": True,  # The F-score of predicted tokens versus the gold answer
#             "HasAns_exact": True,  # Exact match (the normalized answer exactly match the gold answer)
#             "HasAns_f1": True,  # The F-score of predicted tokens versus the gold answer
#             "NoAns_exact": True,  # Exact match (the normalized answer exactly match the gold answer)
#             "NoAns_f1": True,  # The F-score of predicted tokens versus the gold answer
#             "best_exact": True,  # Best exact match (with varying threshold)
#             "best_f1": True,  # Best F1 (with varying threshold)
#         }
