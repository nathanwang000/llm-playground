from langchain.document_loaders import TextLoader, PyPDFLoader
from pprint import pformat
from termcolor import colored
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
import openai, os
import logging
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

openai.api_key = os.environ["OPENAI_API_KEY"]
logger = logging.getLogger(__name__)

def create_retry_decorator(max_tries=3, min_seconds=4, max_seconds=10):
    # from langchain's _create_retry_decorator
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_tries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

class ChatBot:
    '''open ai vannilla chatbot'''
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})
    
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    @create_retry_decorator(max_tries=3)
    def execute(self):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages)
        # Uncomment this to print out token usage each time, e.g.
        # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
        # print(completion.usage)
        return completion.choices[0].message.content
    
def get_input_prompt_session(color='ansired'):
    return PromptSession(
        style=Style.from_dict({'prompt':
                               color}),
    )
    
def repl(f,
         input_prompt=">> ",
         output_prompt=":: "):
    '''
    f is the function that will be called on each input
    '''
    # history enabled and handles spacing well
    session = get_input_prompt_session('ansired')
    while True:
        try:
            user_input = session.prompt(message=input_prompt)
            if user_input.strip() == "":
                continue
        except EOFError:
            # Handle Ctrl+D (End of File) to exit the REPL
            break
        except Exception as e:
            # Handle and print any exceptions that occur during evaluation
            print(f"Error: {e}")
        
        ans = f(user_input)
        print(colored(output_prompt, "green"))
        custom_print(ans)

def custom_print(d):
    if type(d) is not dict:
        print(d)
        return
    print('{')
    indent = ' ' * 2
    for k, v in d.items():
        v = str(v).split('\n')
        v_repr = '\n'.join([indent*2 + line for line in v])
        print(f'{indent}{k}: \n{v_repr}')
    print('}')
    
def load_doc(docname):
    if docname.endswith('.txt'):
        return TextLoader(docname)
    elif docname.endswith('.pdf'):
        return PyPDFLoader(docname)
    else:
        raise Exception("unknown file format")

def strip_multiline(text):
    return "\n".join(list(map(lambda x: x.strip(), text.strip().split('\n'))))


