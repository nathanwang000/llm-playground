# import guidance
from guidance import models, gen, user, assistant

# path = "vicuna-7b"
# vicuna = guidance.llms.transformers.Vicuna(path, device="mps")
# mpt = guidance.llms.transformers.MPTChat('mosaicml/mpt-7b-chat', device=1)

# chatgpt = guidance.llms.OpenAI("gpt-3.5-turbo")

# program = guidance(
#     """The best thing about the beach is {{~gen 'best' temperature=0.7 max_token\
# s=7}}""",
#     stream=True,
# )
# print(program(llm=vicuna))

gpt = models.OpenAI("gpt-3.5-turbo")

with user():
    lm = gpt + "Question: Luke has ten balls. He gives three to his brother.\n"
    lm += "How many balls does he have left?\n"
    lm += "Answer: "

with assistant():
    lm += gen(regex="\d+")
