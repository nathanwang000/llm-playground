# import time
import sys
import base64
import datetime
import functools
import glob
import json
import logging
import os
import pprint
import re
import signal
import subprocess
import tempfile
from collections import namedtuple
from collections.abc import Mapping
from typing import List, Set

import openai
import tqdm

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
    )
)

from const import EXCEPTION_PROMPT

# from const import EXCEPTION_PROMPT

# for caching see https://shorturl.at/tHTV4
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_community import GoogleDriveLoader
from langchain_openai import (
    AzureChatOpenAI,
    ChatOpenAI,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
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


def info(message):
    return colored(message, "cyan")


def success(message):
    return colored(message, "green")


def find_last_date_in_dir(dir_path, date_format="%Y-%m-%d"):
    """
    List all directories in dir_path.
    Find the latest date that is not today in the directory.
    Return the directory name with that date.

    Parameters:
    - dir_path (str): The path to the directory to search in.

    Returns:
    - str: The name of the directory with the last date
    """
    # Initialize variables
    last_date = None
    last_dir = None

    # Iterate over the directories in dir_path
    for dir_name in os.listdir(dir_path):
        # Check if the directory is valid
        if os.path.isdir(os.path.join(dir_path, dir_name)):
            # Get the date from the directory name
            try:
                date = datetime.datetime.strptime(dir_name, date_format).date()
            except ValueError:
                continue

            # Check if the date is later than the last_date
            if last_date is None or date > last_date:
                last_date = date
                last_dir = dir_name

    # Return the directory name with the last date
    return last_dir


def pdf2md_vlm(fn: str, output_dir: str = "output_pdf2md", use_azure=False) -> str:
    """
    Convert a PDF file to a markdown string using gpt4o
    flow: pdf->png->md (img_desc)

    Args:
      fn: PDF file name
      output_dir: Directory to save the output markdown file (irrelevant for vlm)
      verbose: Whether to show extracted images

    Returns:
      md file path

    # >>> pdf2md_vlm('../docs/SMI-hypertension-bundle-emergency-checklist.pdf')
    # >>> pdf2md_vlm('../debug_docs/1.pdf')
    # >>> pdf2md_vlm('../debug_docs/2.pdf')
    # >>> pdf2md_vlm('../debug_docs/3.pdf')
    """
    # Get the file name without extension
    filename = os.path.splitext(os.path.basename(fn))[0]
    images = convert_from_path(fn)
    md_txts = []

    os.system(f"mkdir -p {output_dir}/page_contents/")
    combined_md_file_path = f"{output_dir}/{filename}.md"
    if os.path.exists(combined_md_file_path):  # caching
        print(f"skipping creating {combined_md_file_path} as it already exists")
        return combined_md_file_path

    for i, image in enumerate(images):
        md_file_path = f"{output_dir}/page_contents/{filename}_page{i}.md"
        if os.path.exists(md_file_path):
            print(f"skipping {md_file_path} as it already exists")
            md_txts.append(open(md_file_path).read())
            continue

        png_file_path = f"{output_dir}/page_contents/{filename}_page{i}.png"
        image.save(png_file_path, "PNG")
        print(info(f"processing page {i+1}/{len(images)}"))
        print(image)
        bot = ChatVisionBot(
            "transcribe the image as markdown, keeping the structure of the document (e.g., heading and titles); Be sure to describe figures or visuals or embedded images in the format of '![<desc>](fake_url)'; The entire output will be interpreted as markdown, so don't wrap the output in ```markdown``` delimeter",
            use_vision=True,
            use_azure=use_azure,
        )

        md_txt = print_openai_stream(
            bot(
                "",
                [os.path.expanduser(png_file_path)],
            )
        )

        md_txts.append(md_txt)
        # save intermediate md files
        with open(md_file_path, "w") as f:
            f.write(md_txt)

    with open(combined_md_file_path, "w") as f:
        f.write("\n----\n".join(md_txts))

    return combined_md_file_path


def image2md(image_path, use_azure=False):
    """
    return a md file saved
    """
    os.system("mkdir -p image_descs")
    img_name = os.path.basename(image_path).split(".")[0]
    output_fname = f"image_descs/{img_name}.md"
    if not os.path.exists(output_fname):
        bot = ChatVisionBot(stream=True, use_azure=use_azure, use_vision=True)
        print(info(f"Desc of {image_path} as ctxt:"))
        image_desc = print_openai_stream(
            bot(
                "describe the given image in detail: if the image has structure, format it as markdown",
                [image_path],
            )
        )
        print(f"writing to {output_fname}")
        with open(output_fname, "w") as f:
            f.write(image_desc)

    return output_fname


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
            info("time parsing using Azure:"),
            "Don't sent personal info!"
            " use toggle_settings config.use_azure to turn it off",
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
    print(info("parsed time range:"), result)
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
def encode_image_path(image_path):
    """from official openai"""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"


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
        except Exception:
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
                    print(f"\n\n{info(key)}: {chunk[key]}", end="", flush=True)
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
        print(success(output_prompt))
        printf(ans)


# Function to load the document
@functools.cache
def load_doc(
    fname, use_azure, in_memory_load=True, convert_pdf2md=False
) -> List[Document]:
    """
    load the document from the file

    convert_pdf2md: bool indicating whether to convert pdf to md for better parsing

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
            if convert_pdf2md:
                loader = UnstructuredMarkdownLoader(
                    pdf2md_vlm(fname, use_azure=use_azure)
                )
            else:
                loader = PyPDFLoader(fname)
        elif ext in ["txt", "org"]:
            loader = TextLoader(fname)
        elif ext in ["md"]:
            loader = UnstructuredMarkdownLoader(fname)
        elif ext in ["jpeg", "png", "jpg"]:
            output_fname = image2md(fname, use_azure=use_azure)
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
    res = input(info(prompt + "\nwaiting for response: "))
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


def read_from_dir(
    dirname,
    allowed_extensions=[
        "pdf",
        "md",
        "txt",
        "org",
        "png",
        "jpg",
        "jpeg",
    ],
) -> List[str]:
    """
    reucrusively read all files in the directory, if the directory is a file, return the file(s)
    """
    fnames = []
    if os.path.isfile(dirname):
        print(f"treat {dirname} as a file")
        fnames = [dirname]
    elif os.path.isdir(dirname):
        print(f"treat {dirname} as a dir")
        for fn in glob.glob(dirname + "/*"):
            if os.path.isdir(fn):
                fnames.extend(read_from_dir(fn))
            elif fn.split(".")[-1] in allowed_extensions:
                fnames.append(fn)
    else:  # treat it as glob pattern
        print(f"treat {dirname} as a pattern")
        for fn in glob.glob(dirname):
            if fn.split(".")[-1] in allowed_extensions:
                fnames.append(fn)
    return fnames


@functools.cache
def load_split_doc(
    fname: str,
    use_azure: bool,
    chunk_size: int,
    chunk_overlap: int,
    convert_pdf2md: bool,
) -> List[Document]:
    """
    load documents, split the text using recursive character text splitter
    TODO: add document level cache using document changed time and the arguments (save the result to disk)
    """
    print(info("loading"), fname)
    doc = load_doc(fname, use_azure=use_azure, convert_pdf2md=convert_pdf2md)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    splits = text_splitter.split_documents(doc)
    print(fname, info("split into"), len(splits), "chunks")
    return splits


def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string.:"""
    formatted = [
        f"Article Meta data: {pprint.pformat(doc.metadata)}\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n\n".join(formatted)


def partial_wrap(f, *pargs, add_note=True, **pkwargs):
    """
    behaves like functools.partial but inherent f's doc string and specifies
    args supplied
    """
    partial_f = functools.partial(f, *pargs, **pkwargs)

    @functools.wraps(f)
    def _f(*args, **kwargs):
        return partial_f(*args, **kwargs)

    # Modify the docstring
    if add_note:
        additional_note = (
            "\n\nNote:\nThis function has the following arguments supplied:\n"
        )
        for i, arg in enumerate(pargs):
            additional_note += f"    arg[{i}] = {arg}\n"
        for arg, value in pkwargs.items():
            additional_note += f"    {arg} = {value}\n"

        if f.__doc__:
            _f.__doc__ = f.__doc__ + additional_note
        else:
            _f.__doc__ = additional_note

    return _f


def join_list(item, list):
    """list version of "".join(List[str])
    >>> join_list(' ', ['a', 'b', 'c'])
    ['a', ' ', 'b', ' ', 'c']
    """
    res = []
    for i in list:
        res.append(i)
        res.append(item)
    return res[:-1]  # get rid of the last item


def gen_attr_paths_value(
    obj,
    curr_path: List[str],
    seen: Set,
    depth=0,
):
    """
    return the attr (path, value) pairs

    NOTE: curr_path and seen don't have default of their type because they are mutable

    >>> next(gen_attr_paths_value(1, [], set()))
    ([], 1)
    >>> list(gen_attr_paths_value(ShellCompleter(), [], set()))
    [(['commands'], []), (['command_completer', 'words'], []), (['command_completer', 'ignore_case'], True), (['command_completer', 'display_dict'], {}), (['command_completer', 'meta_dict'], {}), (['command_completer', 'WORD'], False), (['command_completer', 'sentence'], False), (['command_completer', 'match_middle'], False), (['command_completer', 'pattern'], None)]
    """
    # see if obj is primtive types
    if not hasattr(obj, "__dict__"):
        yield curr_path, obj
        return

    # non primitive types need to track whether loop
    if id(obj) in seen:
        yield from []
        return
    seen.add(id(obj))

    for attr in obj.__dict__.keys():
        yield from gen_attr_paths_value(
            getattr(obj, attr, None),
            curr_path + [attr],
            seen,
            depth + 1,
        )


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
            # treat non first argument as path to complete
            # complete file path
            text_to_complete = os.path.expanduser(
                " ".join(chunks[1:]).strip(),
            )

            fname = os.path.basename(text_to_complete)
            for path in glob.glob(text_to_complete + "*"):
                cfname = os.path.basename(path)
                if os.path.isdir(path):
                    cfname = cfname + "/"
                yield Completion(cfname, start_position=-len(fname))


class ChatVisionBot:
    """open ai vannilla chatbot"""

    def __init__(
        self,
        system="",
        stop="<|endoftext|>",
        model="gpt-4o",
        stream=True,
        use_azure=False,
        use_vision=False,  # default to chat api to save cost
        max_tokens=3000,
    ):
        self.model = model
        self.system = system
        self.stop = stop
        self.use_azure = use_azure
        self.max_tokens = max_tokens

        if self.use_azure and os.environ.get("AZURE_VISION_API_KEY"):
            if use_vision:
                print(info("using Azure vision api for chatbot creation"))
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
                print(info("using Azure chat api for chatbot creation"))
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
                            "image_url": {
                                "url": encode_image_path(image_url),
                            },
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
