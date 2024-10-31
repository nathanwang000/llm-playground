import json
import os
from termcolor import colored
from pathlib import Path
from openai import OpenAI

client = OpenAI()


def create_clip_name(save_dir: Path):
    idx: int = 0

    def get_next_clip_name():
        nonlocal idx
        idx += 1
        return save_dir / f"{idx-1}.mp3"

    return get_next_clip_name


def utter(text: str, speech_file_path: str, voice="alloy"):
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
    )

    response.stream_to_file(speech_file_path)


if __name__ == "__main__":
    # load discussion text
    discussion = json.load(open("outputs/demo.json"))
    roles = discussion["roles"]
    roles2voice = dict(zip(roles, ["echo", "alloy", "nova"]))
    assert len(roles2voice) == len(roles), "not enough voice actors"

    save_dir = Path(__file__).parent / "outputs" / "demo"
    os.system(f"mkdir -p {save_dir}")
    get_next_clip_name = create_clip_name(save_dir)

    # conversation starts
    role = "moderator"
    text = f"Hi, I'm {roles2voice['moderator']}, the moderator. Today, {' and '.join(roles[:-1])} will discuss on the topic of {discussion['topic']}"
    print(f"{colored(role, 'red')}: {text}")
    utter(text, get_next_clip_name(), voice=roles2voice[role])

    for item in discussion["hist"]:
        role = item["role"]
        text = item["response"]
        print(f"{colored(role, 'red')}: {text}")
        utter(text, get_next_clip_name(), voice=roles2voice[role])
