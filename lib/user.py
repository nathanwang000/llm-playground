# import time
# import sys
import functools
import yaml
import json
import os
import datetime
import pprint
import re
import subprocess
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any


from const import EXCEPTION_PROMPT
from evaluation import chat_eval

# for caching see https://shorturl.at/tHTV4
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import Chroma
from langchain_openai import (
    AzureOpenAIEmbeddings,
    OpenAIEmbeddings,
)
from termcolor import colored
from utils import (
    ChatVisionBot,
    info,
    success,
    print_openai_stream,
    format_docs,
    load_doc,
    read_from_dir,
    run_subprocess_with_interrupt,
    strip_multiline,
    load_split_doc,
    human_llm,
    ShellCompleter,
    flatten,
    smap,
    parse_diary_entries,
    parse_time_range_from_query,
    find_last_date_in_dir,
)


#### users of llm
@dataclass
class UserConfig:
    chat: bool = False
    model: str = "gpt-4o"
    # debug asking human
    human: bool = False
    show_context: bool = False
    fnames: set = field(default_factory=lambda: set())
    # use azure api
    use_azure: bool = False
    # convert pdf to md
    convert_pdf2md: bool = False


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
        config: UserConfig = UserConfig(),
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
        if self.config.human:
            self.chatbot = human_llm
        else:
            self.chatbot = ChatVisionBot(
                self._known_action_get_prompt(),
                model=self.config.model,
                use_azure=self.config.use_azure,
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
        self.config.model = model
        self._reset()
        return f"set model to {model}"

    def _known_action_toggle_settings(self, setting_name):
        """
        toggle the setting value, allows setting nested values (separated by '.')

        >>> user = User()
        >>> user.config.use_azure = False
        >>> user._known_action_toggle_settings('config.use_azure')
        'setting config.use_azure from False->True'
        """
        last_name = None
        parent = None
        obj = self
        for name in setting_name.split("."):
            parent = obj
            last_name = name
            obj = getattr(obj, name, None)
            if obj is None:
                print(
                    EXCEPTION_PROMPT,
                    f"{setting_name} does not exist",
                )
                return

        if not isinstance(obj, bool):
            print(
                EXCEPTION_PROMPT,
                f"{setting_name} is not bool, its value is {obj}",
            )
            return
        setattr(parent, last_name, not obj)
        self._reset()
        return f"setting {setting_name} from {obj}->{not obj}"

    def _known_action_change_user_config_yaml(self, fname):
        """change self.config to point to fname"""
        self.config = UserConfig(**yaml.safe_load(open(fname)))
        self._reset()

    def _known_action_show_settings(self, *args, **kwargs):
        """show the current settings of the chatbot"""
        return pprint.pformat(
            {k: v for k, v in self.__dict__.items() if k != "known_actions"}
        )

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
            self.config.fnames.add(dirname)
            return self._known_action_ls_ctx_files()

        # expand user as well as escape back whitespace
        dirname = os.path.expanduser(dirname).replace("\ ", " ")
        fnames = read_from_dir(dirname)
        print("found", len(fnames), "files in", dirname)
        print(fnames)
        self.config.fnames.update(fnames)
        return self._known_action_ls_ctx_files()

    def _known_action_rm_files(self, pattern):
        r"""
        remove files in the current chatbot,
        pattern is regex patterns to match the files

        doctest
        >>> user = User()
        >>> user.config.fnames = {"file1.txt", "file2.txt", "file3.txt"}
        >>> user._known_action_rm_files("file[1-2].txt")
        2 files matched the pattern file[1-2].txt : ['file1.txt', 'file2.txt']
        'file3.txt'
        >>> user._known_action_rm_files(".*")
        1 files matched the pattern .* : ['file3.txt']
        ''
        """

        # match pattern to self.fnames
        matched_files = set()
        for fname in self.config.fnames:
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
        self.config.fnames -= matched_files

        return self._known_action_ls_ctx_files()

    def _known_action_ls_ctx_files(self, *args, **kwargs):
        """list all files used in context in the current chatbot"""
        return "\n".join(self.config.fnames)

    def get_completer(self):
        """return autocompleter the current text with the prompt toolkit package"""
        return ShellCompleter(self.known_actions.keys())

    def get_context(self, question: str) -> (str, Any):
        """return the context of the question,
        and meta_data for the answer"""
        raise NotImplementedError(
            "get_context method not implemented, should be implemented in the subclass"
        )

    def rag_call(self, prompt: str):
        context, meta_data = chat_eval(
            self.get_context,
            use_azure=self.config.use_azure,
        )(prompt)
        print(
            info("src of retrieved context:"),
            pprint.pformat(meta_data),
        )

        if not context:
            print(EXCEPTION_PROMPT, "no context found")
        prompt = f"Context: {context}\n\nQuestion: {prompt}"
        print(info("done retrieving relevant context"))

        if self.config.show_context:
            print(info("Context:\n"), context)

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
        return result

    def __call__(self, prompt: str):
        """
        chat with the user
        """

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
                print(info(f"executing bot command {k} {v}"))
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
            print(EXCEPTION_PROMPT, e, info("asking llm"))
            result = self.rag_call(prompt)

        if not self.config.chat:
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
        diary_fn: str,
        profile_fn: str,
        config: UserConfig = UserConfig(),
    ):
        self.profile = load_doc(
            profile_fn,
            use_azure=config.use_azure,
            convert_pdf2md=config.convert_pdf2md,
        )[0].page_content
        self.tone = "casual"
        super().__init__(config)
        # add diary_fn to fnames
        self._known_action_add_files(diary_fn)

    def _known_action_change_tone(self, *args, **kwargs) -> str:
        """change the tone of the chatbot to be more serious or more casual"""
        if len(args) == 0 or args[0].strip() == "":
            return self.tone
        self.tone = args[0]
        return self.tone

    def _known_action_generate_report(self, *args, **kwargs) -> str:
        """generate a report based on the diary since last time
        the report was generated; report saved in <save_dir>/<date>/report.md"""

        save_dir = "logs"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            os.system(
                f'mkdir -p {save_dir}/None; echo "no previous report" > {save_dir}/None/report.md'
            )

        instruction = (
            "how's my last 2 weeks and suggestion for improvement"
            if (len(args) == 0 or not args[0])
            else args[0]
        )
        last_date = find_last_date_in_dir(save_dir)
        today = datetime.datetime.today().date()
        output_path = f"{save_dir}/{today}/report.md"
        if str(last_date) == str(today):
            return f"report already generated today, check {output_path}"

        os.system(f"mkdir -p {save_dir}/{today}")
        # generate the report
        result = self.rag_call(instruction)
        if self.chatbot.stream:
            result = print_openai_stream(result)
        with open(output_path, "w") as f:
            f.write(result)
        return f'report (instr: "{instruction}") saved in {output_path}'

    def _known_action_welcome(self, *args) -> str:
        """
        a piece of text to show when the user starts the program
        """
        message = subprocess.check_output(["figlet", "Know thyself"]).decode()
        inspire = info(
            "You are creative, openminded, and ready to learn new things about this absurd world!"
        )
        quote = (
            success("Random words of wisdom:\n\n")
            + subprocess.check_output(["fortune"]).decode()
        )
        reminder = "\n".join(
            [
                success("Ideas to try:\n"),
                "- learn a new emacs (c-h r) or python trick",
                "- update cheatsheet about me: https://shorturl.at/ltwKW",
                "- my work items are in https://shorturl.at/HLP59",
                "- go out for a walk and not dwell on unchangeable event",
            ]
        )
        user_prompt = "\n".join(
            [
                success("You may wanna ask:\n"),
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

    def get_context(self, question) -> (str, Any):
        """specific retriever for diary"""
        diary = format_docs(
            list(
                flatten(
                    smap(
                        functools.partial(
                            load_doc,
                            use_azure=self.config.use_azure,
                            in_memory_load=True,
                            convert_pdf2md=self.config.convert_pdf2md,
                        ),
                        self.config.fnames,
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

            # fixme: maybe use self.model, but gpt3.5 doesn't work good enough
            s_date, e_date = parse_time_range_from_query(
                question, use_azure=self.config.use_azure
            )
            # test if each entry is relevant to the time in the question
            entries = [e for e in entries if s_date <= e["date"] <= e_date]
        except Exception as e:
            entries = sorted(entries, key=itemgetter("date"), reverse=True)[:7]
            print(
                EXCEPTION_PROMPT,
                e,
                "in diary get_context, using last 7 entries sorted by date",
            )
            print([e["date"].strftime("%m/%d/%Y") for e in entries])

        return "\n\n".join(
            [e["entry"] for e in entries]
        ), "meta data not implemented yet"


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
        config: UserConfig = UserConfig(),
    ):
        super().__init__(config)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # default
        self.max_n_context = 3

    def _known_action_welcome(self, *args) -> str:
        """
        a piece of text to show when the user starts the program
        """
        message = subprocess.check_output(["figlet", "Ask Docs"]).decode()
        inspire = info(
            "You are creative, openminded, and ready to learn new things about this absurd world!"
        )
        quote = (
            success("Random words of wisdom:\n\n")
            + subprocess.check_output(["fortune"]).decode()
        )
        reminder = "\n".join(
            [
                success("Ideas to try:\n"),
                "- Explaination hypothesis: A model's true test of generality lies in its ability to eloquently convey its insights to minds beyond its own.",
                "- Idea: generate explaination to help another model increase its performance -> in turn increase the performance of the first model",
                "- my work items are in https://shorturl.at/HLP59",
            ]
        )
        user_prompt = "\n".join(
            [
                success("You may wanna ask:\n"),
                "- what should I research next?",
                "- insight from my research diary last week?",
                "- add to occasion ideas for event planning: https://shorturl.at/gzQ17",
            ]
        )
        other_tips = "\n".join(
            [
                success("To add google drive access:"),
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

    def get_context(self, question) -> (str, Any):
        """return the context of the question"""
        if not self.config.fnames:
            return ""

        # Load, chunk and index the contents: load doc + chunk and embeddings are cached
        # TODO: use pmap later for multiprocessing, after figuring out how to cache in mp
        docs = list(
            flatten(
                smap(
                    functools.partial(
                        load_split_doc,
                        use_azure=self.config.use_azure,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        convert_pdf2md=self.config.convert_pdf2md,
                    ),
                    self.config.fnames,
                )
            )
        )
        store = LocalFileStore("./cache/")

        # get client: TODO refactor
        if self.config.use_azure and os.environ.get("AZURE_CHAT_API_KEY"):
            print(
                info("Azure chat api for embedding:"),
                "Don't sent personal info!"
                " use toggle_settings config.use_azure to turn it off",
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
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=cached_embedder,
        )
        # print(info("vector store docs"), docs)

        # Retrieve and generate using the relevant snippets of the blog.
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.max_n_context},
        )

        docs = retriever.invoke(question)
        # print(info("retrieved docs"), docs)
        return format_docs(docs), [doc.metadata for doc in docs]
