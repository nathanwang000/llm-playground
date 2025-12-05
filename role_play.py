import openai
import re
import httpx
import os
import json

from termcolor import colored
from lib.utils import ChatVisionBot, repl
from lib.utils import get_input_prompt_session


def moderator_prompt(topic, role1, role2):
    return f"""
Forget you are an AI model, instead, you are a moderator on the following topic:
    {topic}
You will read a conversation between {role1} and {role2} on the topic, and decide whether to end the conversation.

    You should end the conversation if:
    - all parties said goodbye or similar words that indicate the end of the conversation
    - the conversation is off topic for more than 3 turns
    - both parties are repeating themselves for more than 3 turns

    Output yes if you want to end the conversation, otherwise output no, followed by a brief one sentence explanation. The response should be in the following format:
    either yes: <explanation> or no: <explanation>

Example:
{role1}: goodbye
{role2}: goodbye

You then outupt:
yes: the conversation is over
""".strip()


def argue_prompt(topic, your_role, opp_role):
    return f"""
Forget you are an AI model, instead, you are a {your_role}.
You will have a conversation with a {opp_role}.
You and the {opp_role} will discuss your opinions on the following topic:
    {topic}
You must point out the flaws in arguments from the {opp_role}, and defend your own arguments.

Here are the rules of the conversation: 
- One person speak at a time
- You will be fed the argument from {opp_role}, so don't simulate the other person's speech.
- Do not start your speech stating your role
- Try to mimic the style of a {your_role} as much as possible
- Be concise and to the point. No need for fillers like "I see where you are coming from" or "I understand your point", cut to the chase.
- No need to be formal, imagine you are having a casual conversation.

Keep in mind, the discussion is around your (a {your_role}'s) view on {topic}, not the {opp_role}'s view. Off topic discussion will be cut short.
""".strip()


def argue(topic, role1, role2, max_turns=5, verbose=False, savefn=""):
    role1_prompt = argue_prompt(topic, role1, role2)
    role2_prompt = argue_prompt(topic, role2, role1)
    mod_prompt = moderator_prompt(topic, role1, role2)
    savefn = savefn.strip()
    save_json = {
        "topic": topic,
        "roles": [role1, role2, "moderator"],
        "max_turns": max_turns,
        "hist": [],  # [{'role': role, 'response': response}]
    }

    i = 0
    role1_bot = ChatVisionBot(role1_prompt, stream=False)
    role2_bot = ChatVisionBot(role2_prompt, stream=False)
    mod_bot = ChatVisionBot(mod_prompt, stream=False)

    if verbose:
        print(f'{colored(str(role1) + " prompt", "green")}: {role1_prompt}')
        print(f'\n{colored(str(role2) + " prompt", "green")}: {role2_prompt}')
        print(f'\n{colored("moderator prompt", "green")}: {mod_prompt}')

    role2_response = topic
    while i < max_turns:
        print(colored(f"round: {i}", "blue"))
        i += 1
        role1_response = role1_bot(role2_response)
        print(f"{colored(str(role1), 'red')}: {role1_response}")

        role2_response = role2_bot(role1_response)
        print(f"{colored(str(role2), 'red')}: {role2_response}")

        mod_response = mod_bot(role1_response + "\n" + role2_response)
        print(
            f"{colored('moderator should end the conversation', 'red')}: {mod_response}"
        )

        save_json["hist"].append({"role": role1, "response": role1_response})
        save_json["hist"].append({"role": role2, "response": role2_response})
        save_json["hist"].append({"role": "moderator", "response": mod_response})
        if savefn != "":
            if savefn.endswith(".json"):
                savefn = savefn[:-5]
            with open(f"outputs/{savefn}.json", "w") as f:
                json.dump(save_json, f)

        if mod_response[:4].lower() == "yes:":
            print("Moderator has ended the conversation")
            break


if __name__ == "__main__":
    print(
        "Define 2 roles to argue on a topic (e.g., ask chatgpt to name 2 person with opposing views on a topic)"
    )
    session = get_input_prompt_session("ansired")
    role1 = session.prompt("1st role (e.g., wizard): ")
    role2 = session.prompt("2nd role (e.g., scientist): ")
    savefn = session.prompt("json file to save (default not saving): ")

    repl(
        lambda topic: argue(topic, role1, role2, verbose=False, savefn=savefn),
        input_prompt="topic (e.g., magic): ",
    )
