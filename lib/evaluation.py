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
import os
import re
from typing import Callable

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import numpy as np
from termcolor import colored
from const import EXCEPTION_PROMPT


def get_function_info(func: Callable) -> (str, str):
    """
    Get the code body and docstring of a function
    """
    code_body = inspect.getsource(func)
    docstring = inspect.getdoc(func)
    return code_body, docstring


def human(prompt):
    """Prompt the user for input and return the response as an AIMessage; for debugging"""
    res = input(colored(prompt.to_string() + "\nwaiting for response: ", "yellow"))
    return AIMessage(res)


def parse_relevance_output(message: AIMessage) -> (float, str):
    """
    Parse the output of check_relevance

    Assummes input message is in the format:
    score: <score between 0 to 1>
    explanation: <explanation>

    Returns: a dict with keys score and explanation
    """
    scores = list(map(float, re.findall(r"score: (.*)", message.content)))
    explanations = re.findall(r"explanation: (.*)", message.content)

    if not scores:
        print(EXCEPTION_PROMPT, f"no scores found in `{message}`, using nan")
        scores = [np.nan]

    if not explanations:
        print(EXCEPTION_PROMPT, f'no explanation found in `{message}`, using ""')
        explanations = [""]
    return {"score": scores[0], "explanation": explanations[0]}


def check_relevance(
    f: Callable[str, str],
    question: str,
    answer: str,
    ai_mode: bool = False,
    model: str = "gpt-3.5-turbo",
    use_azure: bool = False,
):
    """
    Check the relevance of the answer to the question given the function f

    Args:
    - f: function that takes a question and returns an answer
    - question: input to f
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
        "You can also provide a short explanation.\n"
        "----Example output----\n"
        "score: <score between 0 to 1>\n"
        "explanation: <explanation>"
    )
    p = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "user",
                "Question: ```{question}```\n\n"
                "Answer: ```{answer}```\n\n"
                "Remember your instructions: ```{system_prompt}```",
            ),
        ]
    )  # use .pretty_print() to show prompt

    # human for debug
    if not ai_mode:
        llm = human
    elif use_azure and os.environ.get("AZURE_CHAT_API_KEY"):
        print(
            colored(
                "Using AZURE openAI model for checking relevance."
                " (Don't sent personal info!"
                " use toggle_use_azure to turn it off)",
                "yellow",
            )
        )
        azure_endpoint = os.environ.get("AZURE_CHAT_ENDPOINT")
        api_key = os.environ.get("AZURE_CHAT_API_KEY")
        api_version = os.environ.get("AZURE_CHAT_API_VERSION")
        model = os.environ.get("AZURE_CHAT_MODEL")
        llm = AzureChatOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            model_name=model,
        )
    else:
        llm = ChatOpenAI(model_name=model)

    chain = p | llm | parse_relevance_output
    return chain.invoke(
        {"question": question, "answer": answer, "system_prompt": system_prompt}
    )


def test_f(s: str) -> str:
    """Capitalizes the input string"""
    return s.capitalize()


if __name__ == "__main__":
    print(check_relevance(test_f, "some string", "some other string", ai_mode=True))


def chat_eval(f, use_azure=False):
    """
    Decorator for chat evaluation
    """

    @functools.wraps(f)
    def wrapper(question: str) -> str:
        assert isinstance(question, str), "chat eval only takes string input"
        # TODO: check input
        answer = f(question)
        # TODO: check output
        # check relevance | input, output
        print(colored(f"checking relevance of {f.__name__}", "yellow"))
        rel = check_relevance(
            f,
            question,
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
