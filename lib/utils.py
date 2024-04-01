from langchain_community.document_loaders import TextLoader, PyPDFLoader
from pprint import pformat
from termcolor import colored
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter, Completer, Completion
import openai, os, re, glob, shlex
import logging, tempfile
import signal, subprocess
from collections import deque, OrderedDict
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

EXCEPTION_PROMPT = colored('Exception:', 'red')
openai.api_key = os.environ["OPENAI_API_KEY"]
logger = logging.getLogger(__name__)

# Function to encode the image
def encode_image_path(image_path):
    if image_path.startswith('http'):
        return image_path
    import base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_image}"        

def _process_wrapper(output_queue, func, args, kwargs):
    # Execute the function and put the output in the queue
    output = func(*args, **kwargs)
    output_queue.put(output)

def run_multiprocess_with_interrupt(func, *args, **kwargs):
    '''
    run func in a separate process, handle keyboard interrupt to only kill
    the subprocess
    '''
    import multiprocessing

    # Create a multiprocessing Queue to capture the output
    output_queue = multiprocessing.Queue()

    p = multiprocessing.Process(target=_process_wrapper,
                                args=(output_queue, func, args, kwargs))
    
    try:
        p.start()
        p.join()
    except KeyboardInterrupt:
        p.terminate()
        p.join()
        raise KeyboardInterrupt

    # Retrieve the output from the queue
    output = output_queue.get()
    return output
    
def run_subprocess_with_interrupt(command, check=False, *args, **kwargs):
    '''
    handle run subprocess when keyboard interrupt is issued w/o
    killing the parent process
    check: whether to raise exception if return code is not 0
    all other args are passed to subprocess.Popen
    '''
    p = subprocess.Popen(command, *args, **kwargs)
    try:
        p.wait()
    except KeyboardInterrupt as e:
        p.send_signal(signal.SIGINT)
        p.wait()
        raise e
    if check and p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, command)
  
def unquoted_shell_escape(string):
    '''escape shell string without quotes'''
    return re.sub(r'([ \\\'"!$`])', r'\\\1', string)

def create_retry_decorator(max_tries=3, min_seconds=4, max_seconds=10):
    # from langchain's _create_retry_decorator
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_tries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            # retry every exception
            retry_if_exception_type(Exception)
            # retry_if_exception_type(openai.error.Timeout)
            # | retry_if_exception_type(openai.error.APIError)
            # | retry_if_exception_type(openai.error.APIConnectionError)
            # | retry_if_exception_type(openai.error.RateLimitError)
            # | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

def get_input_prompt_session(color='ansired'):
    if not os.environ.get('SHELL_HISTORY'):
        print('no environment variable SHELL_HISTORY found, history disabled')
        history = None
    else:
        history = FileHistory(os.environ['SHELL_HISTORY'])
    return PromptSession(
        style=Style.from_dict({'prompt':
                               color}),
        history=history
    )

def repl(f,
         input_prompt=">> ",
         output_prompt=":: ",
         completer=None):
    '''
    f is the function that will be called on each input
    '''
    # history enabled and handles spacing well
    session = get_input_prompt_session('ansired')
    while True:
        try:
            user_input = session.prompt(message=input_prompt,
                                        auto_suggest=AutoSuggestFromHistory(),
                                        completer=completer,
                                        complete_while_typing=False,
                                        # for control z
                                        enable_suspend=True)
            if user_input.strip() == "":
                continue
        except EOFError:
            # Handle Ctrl+D (End of File) to exit the REPL
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C to cancel the input prompt
            print(EXCEPTION_PROMPT, 'KeyboardInterrupt')
            continue
        except Exception as e:
            # Handle and print any exceptions that occur during evaluation
            print(EXCEPTION_PROMPT, e)
        
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

class ShellCompleter(Completer):
    def __init__(self, commands):
        self.command_completer = WordCompleter(commands, ignore_case=True)

    def get_completions(self, document, complete_event):
        text_before_cursor = document.text_before_cursor
        chunks = text_before_cursor.split()
        endwithWS = re.compile('.*\s$')        
        if len(chunks) <= 1 and not endwithWS.match(text_before_cursor):
            yield from self.command_completer.get_completions(document, complete_event)
        else:
            if endwithWS.match(text_before_cursor):
                text_to_complete = ''
            else:
                text_to_complete = os.path.expanduser(chunks[-1])

            quote = ""
            directory = os.path.dirname(text_to_complete)
            if directory:
                if directory[0] in ["'", '"']:
                    quote = directory[0]
            fname = os.path.basename(text_to_complete)
            for path in glob.glob(text_to_complete + "*"):
                cfname = os.path.basename(path)
                if quote == '':
                    cfname = unquoted_shell_escape(cfname) # escape without quotes
                if os.path.isfile(path):
                    cfname = cfname + quote
                elif os.path.isdir(path):
                    cfname = cfname + "/"
                yield Completion(cfname, start_position=-len(fname))
            
class ChatBot: # this is using the old API
    '''open ai vannilla chatbot'''
    def __init__(self, system="", stop=None):
        self.system = system
        self.messages = []
        self.stop = stop # stop words
        if self.system:
            self.messages.append({"role": "system", "content": system})
    
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def save_chat(self, filename=""):
        '''save chat messages to json file, need to supply filename'''
        import json
        if filename != '':
            with open(filename, "w") as f:
                json.dump(self.messages, f)
                print(f'chat messages saved to {filename}')
        else:
            with tempfile.NamedTemporaryFile(mode='w+',
                                             delete=False) as f:
                json.dump(self.messages, f)
                print(f'Using tmpfile: {f.name}, as no filename is supplied')

    def load_chat(self, filename):
        '''load chat messages from json file'''
        import json
        self.message = json.load(open(filename, "r"))
            
    @create_retry_decorator(max_tries=3)
    def execute(self):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages, stop=self.stop)
        # Uncomment this to print out token usage each time, e.g.
        # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
        # print(completion.usage)
        return completion.choices[0].message.content

class ChatVisionBot: # this is using the new API
    '''open ai vannilla chatbot'''
    def __init__(self, system="", max_tokens=1000, stop='<|endoftext|>', model="gpt-4-1106-vision-preview"):
        self.model = model
        self.system = system
        self.stop = stop
        self.max_tokens = max_tokens
        self.client = openai.OpenAI()
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})
    
    def __call__(self, message, images=None):
        if images is None:
            images = []

        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message,
                },
                *[{
                    "type": "image_url",
                    "image_url": encode_image_path(image_url)                    
                } for image_url in images]
            ],
        })
            
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def save_chat(self, filename=""):
        '''save chat messages to json file, need to supply filename'''
        import json
        if filename != '':
            with open(filename, "w") as f:
                json.dump(self.messages, f)
                print(f'chat messages saved to {filename}')
        else:
            with tempfile.NamedTemporaryFile(mode='w+',
                                             delete=False) as f:
                json.dump(self.messages, f)
                print(f'Using tmpfile: {f.name}, as no filename is supplied')

    def load_chat(self, filename):
        '''load chat messages from json file'''
        import json
        self.message = json.load(open(filename, "r"))
            
    # @create_retry_decorator(max_tries=3)
    def execute(self):
        completion = self.client.chat.completions.create(model=self.model,
                                                         messages=self.messages,
                                                         max_tokens=self.max_tokens,
                                                         stop=self.stop)
        # Uncomment this to print out token usage each time, e.g.
        # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
        # print(completion.usage)
        return completion.choices[0].message.content
    
