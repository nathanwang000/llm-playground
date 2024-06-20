"""
TODO: Evaluation for a RAG system
question -> *retriever* -> context -> *generator* -> answer

For each text node: question, context, answer
We need to sanitize them for safty, hallucination, etc.
- functions to rewrite the text: t = f_rewrite(t)
  - need to be able to register new rewrite functions
- functions to check the text: f_check(t) -> [float, Explanation]
We can generalize this to a NodeEval class

For each edge between nodes: retriever, generator
We need to evaluate the edge for correctness, etc.
- functions to check the edge
  - For retriever: f_context_relevance(q, c) -> [float, Explanation]
  - For generator: f_answer_groundness(c, a) -> [float, Explanation]
  - For question->answer: f_answer_relevance(q, a) -> [float, Explanation]
We can generalize this to an EdgeEval class

TODO: the generalization of EdgeEval is ChatEval
- assume input and output are strings
- can be created as a decorator function
- get context from docstring, function signature, code body, etc.
- check safety of the input: score, explanation, suggestion for rewrite
- check safety of the output: score, explanation, suggestion for rewrite
- check the relevance of ouptput given input (e.g., groundness, answer/context relevance)

# TODO: try giskard (red teaming) and trullm (rag triad package) for the evaluation
"""

import functools
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
from const import EXCEPTION_PROMPT
from termcolor import colored
from utils import ChatVisionBot, info, print_openai_stream


@dataclass
class Query:
    question: str
    image_paths: List[str] = field(default_factory=list)

    def to_string(self):
        if self.question == "" and len(self.image_paths) == 0:
            return "no query provided"
        elif self.question == "" and len(self.image_paths) != 0:
            return "images"
        elif self.question != "" and len(self.image_paths) == 0:
            return self.question
        else:
            return f"input: ```{self.question}``` and images"


def get_function_info(func: Callable) -> (str, str):
    """
    Get the code body and docstring of a function
    """
    code_body = inspect.getsource(func)
    docstring = inspect.getdoc(func)
    return code_body, docstring


def human(prompt: Query) -> str:
    """Prompt the user for input and return the response as an AIMessage; for debugging"""
    res = input(info(prompt + "\nwaiting for response: "))
    return res


def parse_relevance_output(message: str) -> Dict[str, Any]:
    """
    Parse the output of check_relevance

    Assummes input message is in the format:
    score: <score between 0 to 1>
    explanation: <explanation>

    Returns: a dict with keys score and explanation
    """
    scores = list(map(float, re.findall(r"score: (.*)", message)))
    explanations = re.findall(r"explanation: (.*)", message)

    if not scores:
        print(EXCEPTION_PROMPT, f"no scores found in `{message}`, using nan")
        scores = [np.nan]

    if not explanations:
        print(EXCEPTION_PROMPT, f'no explanation found in `{message}`, using ""')
        explanations = [""]
    return {"score": scores[0], "explanation": explanations[0]}


def check_relevance(
    f: Callable[str, str],
    query: Query,
    answer: str,
    ai_mode: bool = False,
    model: str = "gpt-3.5-turbo",
    use_azure: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Check the relevance of the answer to the question given the function f

    Args:
    - f: function that takes a question and returns an answer
    - query: input to f
    - answer: output of f
    - ai_mode: whether to use AI to evaluate the relevance
    - model: the AI model to use if ai_mode is True
    - use_azure: whether to use Azure OpenAI

    Returns:
    - score: float between 0 and 1
    - explanation: str
    """
    # gather f's information
    code_body, docstring = get_function_info(f)

    if docstring == "":
        print(EXCEPTION_PROMPT, "no docstring, using code body")
        docstring = code_body
        # TODO: generate a docstring from code_body in case default is missing

    system_prompt = (
        "Given a function's information\n"
        f"function info: ```\n{docstring}\n```\n"
        "your job is to check the relevance of the answer to the question."
        " Please provide a score between 0 and 1."
        " 0 means not relevant at all, 1 means very relevant.\n"
        "Your response should strictly follow the following fomrat.\n"
        "----Example output----\n"
        "score: <score between 0 to 1>\n"
        "explanation: <explanation>"
    )

    if ai_mode:
        bot = ChatVisionBot(
            system_prompt,
            stream=verbose,
            use_azure=use_azure,
            use_vision=len(query.image_paths) > 0,
        )
    else:
        bot = human

    if verbose:
        print(info("system_prompt:"), system_prompt)

    prompt = f"""Given \nQuery: ```{query.to_string()}```, \nAnswer: ```{answer}```\nRemember your instructions: ```{system_prompt}```"""
    if verbose:
        print(info("user_prompt:"), prompt)
        print(info("check_relevance output:"))
        output = print_openai_stream(bot(prompt, images=query.image_paths))
    else:
        output = bot(prompt, images=query.image_paths)
    return parse_relevance_output(output)


def test_f(s: str) -> str:
    """Capitalizes the input string"""
    return s.capitalize()


def chat_eval(f, use_azure=False):
    """
    Decorator for chat evaluation
    """

    @functools.wraps(f)
    def wrapper(question: str) -> str:
        assert isinstance(question, str), "chat eval currently only takes string input"
        # TODO: check input
        answer = f(question)
        # TODO: check output
        # check relevance | input, output
        print(info(f"Checking relevance of {f.__name__}"))
        rel = check_relevance(
            f,
            Query(question),
            answer,
            ai_mode=True,
            use_azure=use_azure,
        )
        if rel["score"] < 0.5:
            print(
                EXCEPTION_PROMPT,
                f"low relevance score of {rel['score']}, explanation: {rel['explanation']}\n",
            )
        else:
            print(
                colored(
                    f"relevance check passed with score of {rel['score']} because",
                    "green",
                ),
                rel["explanation"],
            )

        return answer

    return wrapper


if __name__ == "__main__":
    print(check_relevance(test_f, "some string", "some other string", ai_mode=True))
