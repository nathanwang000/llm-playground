'''
cmdline with llm

Runs commands in following order
1. try functions defined in prompt
2. try excecute a system command
3. ask llm
'''
import openai
import click
import os, glob
import subprocess
from utils import ChatBot, repl, strip_multiline
from utils import ShellCompleter


class Cmdline:
    system_prompt = '''
You are an assistant to suggest commands to the user. Be concise in your answers.

When users ask for a command available to you, prioritize recommending the following commands:

{tools}

{llm_tools}

When users ask for a command to impress friends or funny, prioritize recommending the following commands:
    
cowsay - this command displays a message in a speech bubble of an ASCII cow.

fortune - this command displays a random message from a database of quotations.

cmatrix - this command displays the matrix movie in your terminal.
    
It's very likely the user may mistype a command. If you think the command is a typo, suggest the correct command.

Example session:

User: list files in the current directory
AI: ls .

User: resettt
AI: reset
    '''
    def __init__(self):
        self._reset()
        
    def _reset(self, *args, **kwargs):
        '''
        reset the current chatbot session
        can be called stating "reset" in the prompt
        '''
        self.known_actions = {
            'reset': self._reset,
            'lstools': self._list_tools,
            'lsllmtools': self._list_llm_tools,
            'getprompt': self._get_prompt,
        }

        self.chatbot = ChatBot(self._get_prompt())

    def _get_prompt(self, *args, **kwargs):
        '''return the prompt of the current chatbot; use this tool when users ask for the prompt
        such as "show me your prompt"'''
        return self.system_prompt.format(tools=self._list_tools(),
                                         llm_tools=self._list_llm_tools())

    def _list_llm_tools(self, *args, **kwargs):
        '''return a list of llm tools'''
        tools = ['When users ask for llm commands, prioritize recommending the following commands:']
        llm_path = os.environ.get('LLM_PATH', None)
        if llm_path:
            for name in glob.glob(os.path.join(llm_path, '*.py')):
                tools.append(f"python {os.path.join(llm_path, name)}")
        else:
            print('LLM_PATH not set')
        return "\n\n".join(tools)
        
    def _list_tools(self, *args, **kwargs):
        '''return a string describing the available tools to the chatbot; list all tools'''
        tools = []
        for k, v in self.known_actions.items():
            tools.append("{}: \n{}".format(k, strip_multiline(v.__doc__)))
        return "\n\n".join(tools)

    def get_completer(self):
        '''return autocompleter the current text with the prompt toolkit package'''
        return ShellCompleter(self.known_actions.keys())
    
    def __call__(self, prompt):
        prompt = prompt.strip()
        prev_directory = os.getcwd()
        # first try known actions
        if prompt in self.known_actions:
            print('executing bot command {}'.format(prompt))
            return self.known_actions[prompt]()

        # then try to execute the command
        try:
            if prompt.startswith("cd "):
                # Extract the directory from the input
                directory = prompt.split("cd ", 1)[1].strip()
                if directory == '-':
                    directory = prev_directory
                else:
                    prev_directory = os.getcwd()
                # Change directory within the Python process
                os.chdir(directory)
                return directory
            else:
                response = subprocess.run(prompt, check=True, shell=True)
        except Exception as e:
            # finally try to ask the chatbot
            return f"{self.chatbot(prompt)}"


@click.command()
@click.option('-r/-R', 'repl_mode',
              show_default=True,
              default=True,
              help='whether to run in repl mode')
@click.option('-q', 'question',
              prompt=True,
              prompt_required=False,
              default="lstools",
              help="optional command line query",
              show_default=True)
def main(repl_mode, question):
    cmdline = Cmdline()
    if not repl_mode or question != "lstools":
        click.echo(cmdline(question))
    else:
        repl(lambda user_input:
             cmdline(user_input),
             completer=cmdline.get_completer())
        
if __name__ == '__main__':
    main()
