from sglang import function, system, user, assistant, gen, set_default_backend, OpenAI

set_default_backend(OpenAI("gpt-3.5-turbo"))

import sglang as sgl

character_regex = r"""\{
    "name": "[\w\d\s]{1,16}",
    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",
    "blood status": "(Pure-blood|Half-blood|Muggle-born)",
    "wand": \{
        "wood": "[\w\d\s]{1,16}",
        "core": "[\w\d\s]{1,16}",
        "length": [0-9]{1,2}\.[0-9]{0,2}
    \}
\}"""


@sgl.function
def harry_potter_gen(s, name):
    s += sgl.user(f"Please describe the character {name} from Harry Potter.")
    s += sgl.assistant(sgl.gen("json", max_tokens=256, regex=character_regex))


## Regular expression is not supported in the OpenAI backend.
# state = harry_potter_gen.run("Hermione Granger")
# character_json = state["json"]
# print(character_json)
