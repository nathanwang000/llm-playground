import os
import sys
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
from termcolor import colored

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lib.utils import ChatVisionBot, print_openai_stream
from lib.const import EXCEPTION_PROMPT


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


def human(prompt: Query) -> str:
    """Prompt the user for input and return the response as an AIMessage; for debugging"""
    res = input(colored(prompt + "\nwaiting for response: ", "yellow"))
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
    query: Query, answer: str, instruction: str, ai_mode=False
) -> Dict[str, any]:
    """
    given (query, answer) and instruction for judging, otuput a score between 0 and 1

    Args:
      ai_mode: whether to use ai (False for debug)
    """
    system_prompt = (
        f"Instruction: ```{instruction}```\n\n"
        """Output a score between 0 to 1 according to the given instruction.\n"""
        "In addition, output a description. "
        "Your response should strictly follow the following fomrat\n"
        "----Example output----\n"
        "score: <score between 0 to 1>\n"
        "explanation: <explanation>"
    )
    if ai_mode:
        bot = ChatVisionBot(system_prompt, stream=True)
    else:
        bot = human
    print("system_prompt:", colored(system_prompt, "green"))

    prompt = f"""Given \nQuery: ```{query.to_string()}```, \nAnswer: ```{answer}```\nRemember your instructions: ```{system_prompt}```"""
    print("user_prompt:", prompt)
    print("check_relevance output:")
    output = print_openai_stream(bot(prompt, images=query.image_paths))
    return parse_relevance_output(output)
