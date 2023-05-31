from langchain.document_loaders import TextLoader, PyPDFLoader
from pprint import pformat
from termcolor import colored
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter, Completer, Completion
import openai, os, re, glob, shlex
import logging, tempfile
from collections import deque, OrderedDict
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

openai.api_key = os.environ["OPENAI_API_KEY"]
logger = logging.getLogger(__name__)

def org_parser(fn):
    '''
    return a dictionary like data, future could explore json
    understand the following lines
    
    comment: ignore content followed by #
    key word pairs: marked by ':'
    structures: '*' and '**' and '***' mark strucutres

    parses into a dictionary like data structure
    '''
    goals = parse_lines(deque(open(fn, 'r').readlines()), 0)
    return goals

def parse_lines(lines, level):
    '''
    given lines of org mode format
    parse it into python ordered dictionary
    '''
    def clean(l):
        r'''
        strip comment and other special marks
    
        >>> clean("\n")
        ''
        >>> clean(" abc # dsfadfadfease")
        'abc'
        '''
        if '#' in l:
            l = l[:l.index('#')]
        return l.strip()
    
    d = OrderedDict()
    while lines:
        l = clean(lines.popleft())
        m = re.match(r"^(\*+)\W(.*)", l)
        if m: # to another level
            stars, name = m.groups()
            if len(stars) > level:
                d[name] = parse_lines(lines, len(stars))
            else:
                lines.appendleft(l)
                return d
        elif ":" in l: # key value pairs
            k, v = [x.strip() for x in l.split(':')]
            d[k] = v
        elif l == "":
            continue
        else:
            d[l] = None
    return d

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
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
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
                                        complete_while_typing=False)
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
            
class ChatBot:
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
    
