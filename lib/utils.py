# import time
# import sys
import datetime
import functools
import glob
import json
import logging
import base64
import os
import pprint
import re
import signal
import subprocess
import tempfile
from collections import namedtuple
from collections.abc import Mapping
from operator import itemgetter
from typing import List

import openai
import tqdm
from const import EXCEPTION_PROMPT
from evaluation import chat_eval

# for caching see https://shorturl.at/tHTV4
from mimetypes import guess_type
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_community import GoogleDriveLoader
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
    AzureOpenAIEmbeddings,
    AzureChatOpenAI,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from termcolor import colored

openai.api_key = os.environ["OPENAI_API_KEY"]
logger = logging.getLogger(__name__)

# Function to parse time range using llm
DateRange = namedtuple("DateRange", ["start", "end"])
DateRange.__repr__ = (
    lambda x: f'{x.start.strftime("%m/%d/%Y")} - {x.end.strftime("%m/%d/%Y")}'
)


def parse_time_range_from_AI_message(message: AIMessage) -> DateRange:
    def format_date(x):
        return datetime.datetime.strptime(x, "%m/%d/%Y")

    start_dates = [
        format_date(d)
        for d in re.findall(r"start date:\s+(\d{1,2}/\d{1,2}/\d{4})", message.content)
    ]
    end_dates = [
        format_date(d)
        for d in re.findall(r"end date:\s+(\d{1,2}/\d{1,2}/\d{4})", message.content)
    ]

    today = datetime.datetime.today()
    if not end_dates:
        print(
            EXCEPTION_PROMPT,
            f"no end date in AI message {message.content}: set end date as today",
        )
        end_dates = [today]
    if not start_dates or start_dates[0] > end_dates[0]:
        print(
            EXCEPTION_PROMPT,
            "no start date or start date later than end date: set start as end-7",
        )
        start_dates = [end_dates[0] - datetime.timedelta(days=today.weekday() + 7)]
    return DateRange(start_dates[0], end_dates[0])


def parse_time_range_from_query(
    query: str, model: str = "gpt-4-turbo", use_azure=False
) -> DateRange:
    p = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
         Given query, parse out the start and end time of the query.
         If not enough information given, assume time range is the last 7 days

         today: {today}
         query: {query}
         start date: <output start date from query>
         end date: <output end date from query>

         ==== Example ===

         today: Thursday, April 9, 2024 10:46 AM
         query: what did I eat last week?
         start date: 4/2/2024
         end date: 4/9/2024
         """,
            ),
        ]
    )
    today = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")

    if use_azure and os.environ.get("AZURE_CHAT_API_KEY"):
        print(
            colored(
                "Using AZURE openAI model for time parsing"
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
    chain = p | llm | parse_time_range_from_AI_message

    result = chain.invoke({"query": query, "today": today})
    print(colored("parsed time range:", "yellow"), result)
    return result


def parse_diary_entries(diary_txt) -> List[dict]:
    r"""
    Parse diary_txt into chunks marked by each date.
    Assume entry format: "* date: something".
    Parse into [{'date': ..., 'entry': "* date: 4/10/2022\n..."}, ...]

    docstring test:
    >>> diary_txt = '''
    ... * Date: 4/10/2022
    ... I had a great day today.
    ... * Date: 4/11/2022
    ... I had a bad day today.
    ... '''
    >>> parse_diary_entries(diary_txt)
    [{'date': datetime.datetime(2022, 4, 10, 0, 0), 'entry': '* Date: 4/10/2022\nI had a great day today.\n'}, {'date': datetime.datetime(2022, 4, 11, 0, 0), 'entry': '* Date: 4/11/2022\nI had a bad day today.\n\n'}]
    """
    entries = []
    current_entry = None

    for line in diary_txt.split("\n"):
        # Assume date is in the format "* date: 4/10/2022" or "* Date: 04/10/2022"
        date_match = re.search(r"\d{1,2}/\d{1,2}/\d{4}", line)
        if date_match and line.split(":")[0] in ["* Date", "* date"]:
            if current_entry is not None:
                entries.append(current_entry)

            date_entry = date_match.group(0)
            current_entry = {
                "date": datetime.datetime.strptime(date_entry, "%m/%d/%Y"),
                "entry": line + "\n",
            }
        elif current_entry is not None:
            current_entry["entry"] += line + "\n"

    if current_entry is not None:
        entries.append(current_entry)

    return entries


# Function to encode the image
def encode_image_path_openai(image_path: str) -> str:
    if image_path.startswith("http"):
        return image_path

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_image}"


def encode_image_path_azure(image_path: str) -> str:
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found
    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"


def encode_image_path(image_path: str, use_azure: bool = False) -> str:
    print("encoding image base64 use_azure", use_azure)
    if use_azure:
        return encode_image_path_azure(image_path)
    else:
        return encode_image_path_openai(image_path)


# Function to run subprocess
def _process_wrapper(output_queue, func, args, kwargs):
    # Execute the function and put the output in the queue
    output = func(*args, **kwargs)
    output_queue.put(output)


def run_multiprocess_with_interrupt(func, *args, **kwargs):
    """
    run func in a separate process, handle keyboard interrupt to only kill
    the spawned process
    """
    import multiprocessing

    # multiprocessing.set_start_method("fork")
    # print("done forking")

    # Create a multiprocessing Queue to capture the output
    output_queue = multiprocessing.Queue()

    p = multiprocessing.Process(
        target=_process_wrapper, args=(output_queue, func, args, kwargs)
    )

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
    """
    handle run subprocess when keyboard interrupt is issued w/o
    killing the parent process
    check: whether to raise exception if return code is not 0
    all other args are passed to subprocess.Popen
    """
    p = subprocess.Popen(command, *args, **kwargs)
    try:
        p.wait()
    except KeyboardInterrupt as e:
        p.send_signal(signal.SIGINT)
        p.wait()
        raise e
    if check and p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, command)


def run_with_interrupt(f, *args, **kwargs):
    """
    run the function with keyboard interrupt handling,
    sometimes f handles the interrupt itself w/o raising exception again
    this function will intercept the signal
    """

    def handler(signum, frame):
        # raise a custom exception so f won't handle it
        raise Exception("caught signal interruption in run_exception")

    signal.signal(signal.SIGINT, handler)
    return f(*args, **kwargs)


def unquoted_shell_escape(string):
    """escape shell string without quotes"""
    return re.sub(r'([ \\\'"!$`])', r"\\\1", string)


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


# Function to get input prompt session
def get_input_prompt_session(color="ansired"):
    if not os.environ.get("SHELL_HISTORY"):
        print("no environment variable SHELL_HISTORY found, history disabled")
        history = None
    else:
        history = FileHistory(os.environ["SHELL_HISTORY"])
    return PromptSession(style=Style.from_dict({"prompt": color}), history=history)


# Function to print the stream
def print_openai_stream(ans) -> str:
    # from https://cookbook.openai.com/examples/how_to_stream_completions
    if type(ans) is not openai.Stream:
        custom_print(ans)
        return

    collected_chunks = []
    collected_messages = []
    # start_time = time.time()
    for chunk in ans:
        # chunk_time = time.time() - start_time
        collected_chunks.append(chunk)  # save the event response
        # extract the message
        if not len(chunk.choices):
            continue
        try:
            chunk_message = chunk.choices[0].delta.content
        except Exception as e:
            m = chunk.choices[0].messages
            if len(m) and "delta" in m[0] and "content" in m[0]["delta"]:
                chunk_message = m[0]["delta"]["content"]  # for azure
            else:
                continue
        if chunk_message is not None:
            collected_messages.append(chunk_message)  # save the message
            print(chunk_message, end="", flush=True)
        # print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text
    print()
    return "".join(collected_messages)


def print_langchain_stream(ans) -> str:
    # from https://python.langchain.com/docs/use_cases/question_answering/streaming/
    output = {}  # for dict chunk
    curr_key = None
    for chunk in ans:
        if isinstance(chunk, Mapping):
            for key in chunk:
                if key not in output:
                    output[key] = chunk[key]
                else:
                    output[key] += chunk[key]
                if key != curr_key:
                    print(
                        f"\n\n{colored(key, 'green')}: {chunk[key]}", end="", flush=True
                    )
                else:
                    print(chunk[key], end="", flush=True)
                curr_key = key
        else:
            print(chunk, end="", flush=True)
    print()
    return str(output)


def custom_print(d):
    """
    custom print for dictionary, to print nested dictionary
    """
    if not isinstance(d, dict):
        print(d)
        return
    print("{")
    indent = " " * 2
    for k, v in d.items():
        v = str(v).split("\n")
        v_repr = "\n".join([indent * 2 + line for line in v])
        print(f"{indent}{k}: \n{v_repr}")
    print("}")


# Function to run the REPL
def repl(
    f, input_prompt=">> ", output_prompt=":: ", completer=None, printf=custom_print
):
    """
    f is the function that will be called on each input
    """
    # history enabled and handles spacing well
    session = get_input_prompt_session("ansired")
    while True:
        try:
            user_input = session.prompt(
                message=input_prompt,
                auto_suggest=AutoSuggestFromHistory(),
                completer=completer,
                complete_while_typing=False,
                # for control z
                enable_suspend=True,
            )
            if user_input.strip() == "":
                continue
        except EOFError:
            # Handle Ctrl+D (End of File) to exit the REPL
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C to cancel the input prompt
            print(EXCEPTION_PROMPT, "KeyboardInterrupt")
            continue
        except Exception as e:
            # Handle and print any exceptions that occur during evaluation
            print(EXCEPTION_PROMPT, e)

        ans = f(user_input)
        print(colored(output_prompt, "green"))
        printf(ans)


# Function to load the document
@functools.cache
def load_doc(fname, use_azure, in_memory_load=True) -> List[Document]:
    """
    load the document from the file
    # TODO: change naive load to using lazy load to save memory
    # https://python.langchain.com/docs/modules/data_connection/document_loaders/custom/
    """
    # TODO: currently only support my umich google drive, need to provide tool to load different credentials
    # TODO: when adding google folder, list out individual files in the folder
    # google drive loader:
    # follow https://developers.google.com/drive/api/quickstart/python # run the code to get token.json
    # remember to change scopes to .../auth/drive and auth/docs in the app setting and when running the script
    # save token.json to ../../../diary/secrets/token.json relative to current file path; TODO: use env var
    # follow https://developers.google.com/drive/api/quickstart/python
    token_path = f"{os.path.dirname(__file__)}/../../../diary/secrets/token.json"
    if fname.startswith("https://drive.google.com"):
        # see https://python.langchain.com/docs/integrations/document_loaders/google_drive/
        # e.g., https://drive.google.com/drive/folders/1shC6DvfVAF4LCc5VZdMFoSkafpZv3p0O
        loader = GoogleDriveLoader(
            folder_id=fname.split("/")[-1],
            token_path=token_path,
            # Optional: configure whether to recursively fetch files from subfolders. Defaults to False.
            recursive=True,
        )
    elif fname.startswith("https://docs.google.com"):
        # e.g., https://docs.google.com/document/d/1bfaMQ18_i56204VaQDVeAFpqEijJTgvurupdEDiaUQw/edit
        print(fname)
        print(fname.split("/")[-2])
        loader = GoogleDriveLoader(
            document_ids=[fname.split("/")[-2]],
            token_path=token_path,
        )
    else:
        ext = fname.split(".")[-1]
        if ext == "pdf":
            loader = PyPDFLoader(fname)
        elif ext in ["txt", "md", "org"]:
            loader = TextLoader(fname)
        elif ext in ["jpeg", "png", "jpg"]:
            os.system("mkdir -p image_descs")
            img_name = os.path.basename(fname).split(".")[0]
            output_fname = f"image_descs/{img_name}.md"
            if not os.path.exists(output_fname):
                bot = ChatVisionBot(stream=True, use_azure=use_azure)
                print(colored(f"Desc of {fname} as ctxt:", "yellow"))
                # image_desc = bot("describe the image", [fname])
                image_desc = print_openai_stream(
                    bot(
                        "describe the given image in detail: if the image has structure, format it as markdown",
                        [fname],
                    )
                )
                print(f"writing to {output_fname}")
                with open(output_fname, "w") as f:
                    f.write(image_desc)

            loader = UnstructuredMarkdownLoader(output_fname)
        else:
            raise ValueError(f"unsupported file extension {ext}")

    if in_memory_load:
        return loader.load()
    else:
        return loader


def strip_multiline(text):
    return "\n".join(list(map(lambda x: x.strip(), text.strip().split("\n"))))


def human_llm(prompt):
    if not isinstance(prompt, str):
        prompt = prompt.to_string()
    res = input(colored(prompt + "\nwaiting for response: ", "yellow"))
    return AIMessage(res)


def flatten(lists):
    # check if iterable
    if isinstance(lists, (list, tuple)):
        for el in lists:
            yield from flatten(el)
    else:
        yield lists
    return


def pmap(f, inputs):
    """inputs are iterable of input to the process function f,
    doesn't work with anonymous functions"""
    import multiprocessing

    num_processes = multiprocessing.cpu_count()
    print(num_processes, "cpus for parrallel map using multiprocesses")
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm.tqdm(pool.imap(f, inputs), total=len(inputs)))

    return results


def smap(f, inputs):
    """
    serial version of pmap with progress bar
    """
    return list(tqdm.tqdm(map(f, inputs), total=len(inputs)))


def read_from_dir(dirname, allowed_extensions=["pdf", "md", "txt", "org"]) -> List[str]:
    """
    reucrusively read all files in the directory, if the directory is a file, return the file
    """
    if os.path.isfile(dirname):
        return [dirname]

    fnames = []
    for fn in glob.glob(dirname + "/*"):
        if os.path.isdir(fn):
            fnames.extend(read_from_dir(fn))
        elif fn.split(".")[-1] in allowed_extensions:
            fnames.append(fn)
    return fnames


@functools.cache
def load_split_doc(
    fname: str, use_azure: bool, chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """
    load documents, split the text using recursive character text splitter
    TODO: add document level cache using document changed time and the arguments (save the result to disk)
    """
    print("loading", fname)
    doc = load_doc(fname, use_azure=use_azure)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    splits = text_splitter.split_documents(doc)
    print(fname, "split into", len(splits), "chunks")
    return splits


def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string.:"""
    formatted = [
        f"Article Meta data: {pprint.pformat(doc.metadata)}\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n\n".join(formatted)


class ShellCompleter(Completer):
    def __init__(self, commands=None):
        if commands is None:
            commands = []
        self.commands = commands
        self.command_completer = WordCompleter(commands, ignore_case=True)

    def cascade_match(self, text: str, commands: List[str]):
        """
        text: str, the text to match
        commands: list of str, the vocabulary to match against

        cascade rules for matching, if fail on one level, fall back
        to next level, return results
        a) starts with the text
        b) regular expression
        c) contains the text
        d) contains the text (case insensitive)
        e) todo: semantic search
        """
        rules = [
            lambda x: x.startswith(text),
            lambda x: re.match(text, x),
            lambda x: text in x,
            lambda x: text.lower() in x.lower(),
        ]
        for rule in rules:
            res = [x for x in commands if rule(x)]
            if len(res) != 0:
                return res

        return []

    def get_completions(self, document, complete_event):
        text_before_cursor = document.text_before_cursor
        chunks = text_before_cursor.split()
        endwithWS = re.compile(".*\s$")
        if len(chunks) == 0 and not endwithWS.match(text_before_cursor):
            # complete command
            yield from self.command_completer.get_completions(document, complete_event)
        elif len(chunks) == 1 and not endwithWS.match(text_before_cursor):
            # cascade match (allow multiple levels of match like starts with, contains, etc.)
            results = self.cascade_match(chunks[-1], self.commands)
            yield from [Completion(r, start_position=-len(chunks[-1])) for r in results]
        else:
            # complete file path
            if endwithWS.match(text_before_cursor):
                text_to_complete = ""
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
                if quote == "":
                    cfname = unquoted_shell_escape(cfname)  # escape without quotes
                if os.path.isfile(path):
                    cfname = cfname + quote
                elif os.path.isdir(path):
                    cfname = cfname + "/"
                yield Completion(cfname, start_position=-len(fname))


class ChatVisionBot:
    """open ai vannilla chatbot"""

    def __init__(
        self,
        system="",
        stop="<|endoftext|>",
        model="gpt-4-turbo",
        stream=True,
        use_azure=False,
        max_tokens=3000,
    ):
        self.model = model
        self.system = system
        self.stop = stop
        self.use_azure = use_azure
        self.max_tokens = max_tokens

        if self.use_azure and os.environ.get("AZURE_VISION_API_KEY"):
            print(
                colored(
                    "Using AZURE openAI model for chat "
                    "(Don't sent personal info!"
                    " use toggle_use_azure to turn it off)",
                    "yellow",
                )
            )

            use_vision = True
            if use_vision:
                proxies = {
                    "http": "",
                    "https": "",
                }
                openai.proxy = proxies
                api_base = os.environ.get("AZURE_VISION_API_BASE")
                api_key = os.environ.get("AZURE_VISION_API_KEY")
                deployment_name = os.environ.get("AZURE_VISION_DEPLOYMENT_NAME")
                api_version = os.environ.get("AZURE_VISION_API_VERSION")

                self.client = openai.AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    base_url=f"{api_base}openai/deployments/{deployment_name}/extensions",
                )
            else:  # text chat only model
                azure_endpoint = os.environ.get("AZURE_CHAT_ENDPOINT")
                api_key = os.environ.get("AZURE_CHAT_API_KEY")
                api_version = os.environ.get("AZURE_CHAT_API_VERSION")
                self.model = os.environ.get("AZURE_CHAT_MODEL")
                self.client = openai.AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version,
                )
        else:
            self.client = openai.OpenAI()
        self.messages = []
        self.stream = stream
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message, images=None):
        if images is None:
            images = []

        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message,
                    },
                    *[
                        {
                            "type": "image_url",
                            "image_url": encode_image_path(image_url, self.use_azure),
                        }
                        for image_url in images
                    ],
                ],
            }
        )

        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def save_chat(self, filename=""):
        """save chat messages to json file, need to supply filename"""
        import json

        if filename != "":
            with open(filename, "w") as f:
                json.dump(self.messages, f)
                print(f"chat messages saved to {filename}")
        else:
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
                json.dump(self.messages, f)
                print(f"Using tmpfile: {f.name}, as no filename is supplied")

    def load_chat(self, filename):
        """load chat messages from json file"""
        import json

        self.message = json.load(open(filename, "r"))

    # @create_retry_decorator(max_tries=3)
    def execute(self):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=self.stream,
            stop=self.stop,
            max_tokens=self.max_tokens,
        )
        if self.stream:
            return completion
        else:
            # Uncomment this to print out token usage each time, e.g.
            # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
            print(completion.usage)
            return completion.choices[0].message.content


#### users of llm
class User:
    system_prompt = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.
    Cite the exact lines from the context that supports your answer.

    If you happen to know the answer outside the provided context,
    clearly indicate that and provide the answer.

    ------------------- beginning of tool descriptions -------------------
    When users ask for chat bot related commands, suggest the following:
    
    {tools}
    """

    def __init__(
        self,
        chat=False,
        model="gpt-4o",
        human=False,
        show_context=False,
        fnames=set(),
        use_azure=False,  # don't trust using azure
    ):
        # shared across Users
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})
        self._reset()

    def _add_hot_keys(self):
        """add hot keys to the known actions"""
        # get hot_keys from env
        hot_keys_fn = os.environ.get("HOT_KEYS_FN", "")
        if not hot_keys_fn:
            print(colored("env var HOT_KEYS_FN not set", "red"))
            return

        hot_keys = json.load(open(hot_keys_fn, "r"))
        for k, v in hot_keys.items():
            if k not in self.known_actions:
                self.known_actions[k] = v

    def _add_known_actions(self, cls):
        """recursively add known actions from the class"""
        for name in dir(cls):
            # or could use cls.__dict__.items() to get the method
            method = getattr(cls, name)
            if callable(method) and name.startswith("_known_action_"):
                action_name = name[len("_known_action_") :]
                if action_name not in self.known_actions:
                    # prefer the method in the subclass
                    self.known_actions[action_name] = functools.partial(method, self)
        for base_cls in cls.__bases__:
            self._add_known_actions(base_cls)

    def _reset(self, *args, **kwargs):
        """
        reset the current chatbot session
        can be called stating "reset" in the prompt
        """
        self.known_actions = {}
        self._add_hot_keys()
        self._add_known_actions(self.__class__)

        # reset chatbot: human for debugging
        if self.human:
            self.chatbot = human_llm
        else:
            self.chatbot = ChatVisionBot(
                self._known_action_get_prompt(),
                model=self.model,
                use_azure=self.use_azure,
            )
            self.known_actions["save_chat"] = self.chatbot.save_chat

    def _known_action_welcome(self, *args) -> str:
        """
        a piece of text to show when the user starts the program
        """
        raise NotImplementedError(
            "_known_action_welcome method not implemented, should be implemented in the subclass"
        )

    def _known_action_set_model(self, model):
        """set the model of the current chatbot"""
        self.model = model
        self._reset()
        return f"set model to {model}"

    def _known_action_show_settings(self, *args, **kwargs):
        """show the current settings of the chatbot"""
        return pprint.pformat(self.__dict__)

    def _known_action_toggle_use_azure(self, *args, **kwargs) -> str:
        """toggle using azure or not"""
        self.use_azure = not self.use_azure
        # need to reset so that chatbot uses the new setting
        self._reset()
        return f"use azure: {self.use_azure}"

    def _known_action_toggle_show_context(self, *args, **kwargs) -> str:
        """toggle showing the context of the question"""
        self.show_context = not self.show_context
        return f"showing context: {self.show_context}"

    def _known_action_get_prompt(self, *args, **kwargs):
        '''return the prompt of the current chatbot; use this tool when users ask for the prompt
        such as "show me your prompt"'''
        return "You are a generic AI. Tell user to implement _known_action_get_prompt in the subclass."

    def _known_action_list_tools(self, *args, **kwargs):
        """return a string describing the available tools to the chatbot; list all tools"""
        tools = []
        for k, v in self.known_actions.items():
            description = strip_multiline(v.__doc__) if v.__doc__ else str(v)
            tools.append("{}: \n{}".format(k, description))
        return "\n\n".join(tools)

    def _known_action_add_files(self, dirname):
        """add a directory to the current chatbot, if given a file, add the file instead"""
        if dirname.startswith("https://"):
            print("adding", dirname)
            self.fnames.add(dirname)
            return self._known_action_ls_ctx_files()

        dirname = os.path.expanduser(dirname)
        fnames = read_from_dir(dirname)
        print("found", len(fnames), "files in", dirname)
        print(fnames)
        self.fnames.update(fnames)
        return self._known_action_ls_ctx_files()

    def _known_action_rm_files(self, pattern):
        r"""
        remove files in the current chatbot,
        pattern is regex patterns to match the files

        doctest
        >>> user = User()
        >>> user.fnames = {"file1.txt", "file2.txt", "file3.txt"}
        >>> user._known_action_rm_files("file[1-2].txt")
        2 files matched the pattern file[1-2].txt : ['file1.txt', 'file2.txt']
        'file3.txt'
        >>> user._known_action_rm_files(".*")
        1 files matched the pattern .* : ['file3.txt']
        ''
        """

        # match pattern to self.fnames
        matched_files = set()
        for fname in self.fnames:
            try:
                if re.match(f"^{pattern}$", fname):
                    matched_files.add(fname)
            except re.error as e:
                print(EXCEPTION_PROMPT, "in regex", pattern, e)

        print(
            len(matched_files),
            "files matched the pattern",
            pattern,
            ":",
            sorted(list(matched_files)),
        )

        # remove the matched files
        self.fnames -= matched_files

        return self._known_action_ls_ctx_files()

    def _known_action_ls_ctx_files(self, *args, **kwargs):
        """list all files used in context in the current chatbot"""
        return "\n".join(self.fnames)

    def get_completer(self):
        """return autocompleter the current text with the prompt toolkit package"""
        return ShellCompleter(self.known_actions.keys())

    def get_context(self, question: str) -> str:
        """return the context of the question"""
        raise NotImplementedError(
            "get_context method not implemented, should be implemented in the subclass"
        )

    def __call__(self, prompt: str):
        prompt = prompt.strip()
        prev_directory = os.getcwd()

        try:
            # first try hot_keys (FIXME: directly calling user input is not secure)
            if prompt.strip() in self.known_actions:
                v = self.known_actions[prompt.strip()]
                if isinstance(v, str):
                    print("Hot Key command:", v)
                    return run_subprocess_with_interrupt(v, check=True, shell=True)

            # then try known actions
            if prompt and prompt.split()[0] in self.known_actions:
                k = prompt.split()[0]
                v = prompt[len(k) :].strip()
                print(f"executing bot command {k} {v}")
                return self.known_actions[k](v)

            # then try to execute the command
            if prompt.startswith("cd "):
                # Extract the directory from the input
                directory = prompt.split("cd ", 1)[1].strip()
                if directory == "-":
                    directory = prev_directory
                else:
                    prev_directory = os.getcwd()
                # Change directory within the Python process
                os.chdir(directory)
                return directory
            else:
                # subprocess start a new process, thus forgetting aliases and history
                # solution: override system command by prepending to $PATH
                # and use shared history file (search chatgpt)

                # handle control-c correctly for child process (o/w kill parent python process)
                # if don't care, then uncomment the following line, and comment out others
                # subprocess.run(prompt, check=True, shell=True)
                return run_subprocess_with_interrupt(prompt, check=True, shell=True)

        except KeyboardInterrupt:
            print(EXCEPTION_PROMPT, "KeyboardInterrupt")  # no need to ask llm
            return None
        except Exception as e:
            print(EXCEPTION_PROMPT, e, colored("asking llm", "yellow"))

            context = chat_eval(
                self.get_context,
                use_azure=self.use_azure,
            )(prompt)
            if not context:
                print(EXCEPTION_PROMPT, "no context found")
            prompt = f"Context: {context}\n\nQuestion: {prompt}"
            print("done retrieving relevant context")

            if self.show_context:
                print(colored("Context:\n", "green"), context)

            # handle c-c correctly, o/w kill parent python process (e.g., self.chatbot(prompt))
            # so far mp based method have pickle issues
            try:
                result = self.chatbot(prompt)
            except KeyboardInterrupt:
                print(
                    EXCEPTION_PROMPT,
                    "Keyboard interrupt when sending to the guru (llm):",
                )
                result = None

        if not self.chat:
            self._reset()
        return result


class DiaryReader(User):
    system_prompt = """
    You are given a user profile and the user's diary entries.
    You will act like you are the user and respond to questions about the user.
    When answering questions, cite your source (with dates if possible).

    Be as helpful as possible, even if the context doesn't provide the answer.
    
    Be concise unless instructed otherwise.

    Response tone: {tone}

    ------------------- beginning of tool descriptions -------------------
    When users ask for chat bot related commands, suggest the following:
    
    {tools}

    -------------------- end of tool descriptions --------------------

    user profile: ```{profile}```
    """

    def __init__(
        self,
        diary_fn,
        profile_fn,
        # needed with base class
        chat=False,
        model="gpt-4-turbo",
        human=False,
        fnames=set(),
        use_azure=False,
    ):
        self.profile = load_doc(profile_fn, use_azure=use_azure)[0].page_content
        self.tone = "less serious"
        super().__init__(
            chat=chat,
            model=model,
            human=human,
            show_context=False,
            fnames=fnames,
            use_azure=use_azure,
        )
        # add diary_fn to fnames
        self._known_action_add_files(diary_fn)

    def _known_action_change_tone(self, *args, **kwargs) -> str:
        """change the tone of the chatbot to be more serious or more casual"""
        if len(args) == 0 or args[0].strip() == "":
            return self.tone
        self.tone = args[0]
        return self.tone

    def _known_action_welcome(self, *args) -> str:
        """
        a piece of text to show when the user starts the program
        """
        message = subprocess.check_output(["figlet", "Know thyself"]).decode()
        inspire = colored(
            "You are creative, openminded, and ready to learn new things about this absurd world!",
            "yellow",
        )
        quote = (
            colored("Random words of wisdom:\n\n", "green")
            + subprocess.check_output(["fortune"]).decode()
        )
        reminder = "\n".join(
            [
                colored("Ideas to try:\n", "green"),
                "- learn a new emacs (c-h r) or python trick",
                "- update cheatsheet about me: https://shorturl.at/ltwKW",
                "- my work items are in https://shorturl.at/HLP59",
                "- go out for a walk and not dwell on unchangeable event",
            ]
        )
        user_prompt = "\n".join(
            [
                colored("You may wanna ask:\n", "green"),
                "- what should I learn next?",
                "- how to improve from last week?",
            ]
        )
        return f"{message}\n{inspire}\n\n{quote}\n{reminder}\n\n{user_prompt}\n"

    def _known_action_get_prompt(self, *args, **kwargs):
        '''return the prompt of the current chatbot; use this tool when users ask for the prompt
        such as "show me your prompt"'''
        return self.system_prompt.format(
            tools=self._known_action_list_tools(), profile=self.profile, tone=self.tone
        )

    def get_context(self, question) -> str:
        """specific retriever for diary"""
        diary = format_docs(
            list(
                flatten(
                    smap(
                        functools.partial(
                            load_doc,
                            use_azure=self.use_azure,
                            in_memory_load=True,
                        ),
                        self.fnames,
                    )
                )
            )
        )

        try:
            try:
                # extract the diary context
                entries = parse_diary_entries(diary)
            except Exception as e:
                print(
                    EXCEPTION_PROMPT,
                    e,
                    "in get_context(), using full entries",
                )
                return diary

            s_date, e_date = parse_time_range_from_query(
                question, use_azure=self.use_azure
            )  # fixme: maybe use self.model, but gpt3.5 doesn't work good enough
            # test if each entry is relevant to the time in the question
            entries = [e for e in entries if s_date <= e["date"] <= e_date]
        except Exception as e:
            entries = sorted(entries, key=itemgetter("date"), reverse=True)[:7]
            print(
                EXCEPTION_PROMPT,
                e,
                "in diary_content_retriever(), using last 7 entries sorted by date",
            )
            print([e["date"].strftime("%m/%d/%Y") for e in entries])

        return "\n\n".join([e["entry"] for e in entries])


class DocReader(User):
    system_prompt = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.
    Cite the exact lines from the context that supports your answer.

    If you know the answer outside the provided context,
    clearly indicate that and provide the answer.

    ------------------- beginning of tool descriptions -------------------
    When users ask for chat bot related commands, suggest the following:
    
    {tools}
    """

    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=200,
        # needed with base class
        chat=False,
        model="gpt-4-turbo",
        human=False,
        fnames=set(),
        use_azure=False,
    ):
        super().__init__(
            chat=chat,
            model=model,
            human=human,
            show_context=False,
            fnames=fnames,
            use_azure=use_azure,
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # default
        self.max_n_context = 3

    def _known_action_welcome(self, *args) -> str:
        """
        a piece of text to show when the user starts the program
        """
        message = subprocess.check_output(["figlet", "Ask Docs"]).decode()
        inspire = colored(
            "You are creative, openminded, and ready to learn new things about this absurd world!",
            "yellow",
        )
        quote = (
            colored("Random words of wisdom:\n\n", "green")
            + subprocess.check_output(["fortune"]).decode()
        )
        reminder = "\n".join(
            [
                colored("Ideas to try:\n", "green"),
                "- Explaination hypothesis: A model's true test of generality lies in its ability to eloquently convey its insights to minds beyond its own.",
                "- Idea: generate explaination to help another model increase its performance -> in turn increase the performance of the first model",
                "- my work items are in https://shorturl.at/HLP59",
            ]
        )
        user_prompt = "\n".join(
            [
                colored("You may wanna ask:\n", "green"),
                "- what should I research next?",
                "- insight from my research diary last week?",
                "- add to occasion ideas for event planning: https://shorturl.at/gzQ17",
            ]
        )
        other_tips = "\n".join(
            [
                colored("To add google drive access:", "green"),
                "- follow https://developers.google.com/drive/api/quickstart/python # run the code to get token.json",
                "- remember to change scopes to .../auth/drive and auth/docs in the app setting and when running the script",
                "- save token.json to secrets/token.json",
                "- go to the googledriveloader file in langchain communit and change the scope accordingly",
            ]
        )
        return f"{message}\n{inspire}\n\n{quote}\n{reminder}\n\n{user_prompt}\n\n{other_tips}\n"

    def _known_action_set_n_context(self, n):
        """set the number of context to show"""
        self.max_n_context = int(n)

    def _known_action_get_prompt(self, *args, **kwargs):
        '''return the prompt of the current chatbot; use this tool when users ask for the prompt
        such as "show me your prompt"'''
        return self.system_prompt.format(tools=self._known_action_list_tools())

    def get_context(self, question) -> str:
        """return the context of the question"""
        if not self.fnames:
            return ""

        # Load, chunk and index the contents: load doc + chunk and embeddings are cached
        # TODO: use pmap later for multiprocessing, after figuring out how to cache in mp
        docs = list(
            flatten(
                smap(
                    functools.partial(
                        load_split_doc,
                        use_azure=self.use_azure,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    ),
                    self.fnames,
                )
            )
        )
        store = LocalFileStore("./cache/")

        # get client: TODO refactor
        if self.use_azure and os.environ.get("AZURE_CHAT_API_KEY"):
            print(
                colored(
                    "Using AZURE openAI model for embedding"
                    " (Don't sent personal info!"
                    " use toggle_use_azure to turn it off)",
                    "yellow",
                )
            )
            azure_endpoint = os.environ.get("AZURE_CHAT_ENDPOINT")
            api_key = os.environ.get("AZURE_CHAT_API_KEY")
            api_version = os.environ.get("AZURE_CHAT_API_VERSION")
            underlying_embeddings = AzureOpenAIEmbeddings(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                model="text-embedding-ada-002",
            )
        else:
            underlying_embeddings = OpenAIEmbeddings()

        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, store, namespace=underlying_embeddings.model
        )
        vectorstore = Chroma.from_documents(documents=docs, embedding=cached_embedder)

        # Retrieve and generate using the relevant snippets of the blog.
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.max_n_context})

        return (retriever | format_docs).invoke(question)
