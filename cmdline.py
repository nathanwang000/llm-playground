'''
cmdline with llm

Runs commands in following order
1. try functions defined in prompt
2. try excecute a system command
3. ask llm
'''
import openai
import click
import os, glob, re
import subprocess
from lib.utils import ChatBot, repl, strip_multiline
from lib.utils import ShellCompleter

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
    
It's very likely the user mistypes a command. If you think the command is a typo, suggest the correct command.

At the end of your response, add a newline and output the command you are suggesting (e.g., command: <executable command>).
    
Example session:

User: list files in the current directory
AI: The following command lists all files in the current directory.
    command: ls

User: rr
AI: Do you mean the "r" command that resets the chatbot session?
    command: r
    '''
    def __init__(self):
        self._reset()
        
    def _reset(self, *args, **kwargs):
        '''
        reset the current chatbot session
        can be called stating "reset" in the prompt
        '''
        self.known_actions = {
            'r': self._reset,
            'l': self._list_tools,
            'll': self._list_llm_tools,
            'p': self._get_prompt,
            'e': self._run_last_command_from_llm,
        }

        self.chatbot = ChatBot(self._get_prompt())

    def _run_last_command_from_llm(self):
        '''run the last command from llm output of the form "command: <executable command>"'''
        command_re = re.compile(r'^[Cc]ommand: (.*)$')
        for message in self.chatbot.messages[::-1]:
            if message['role'] == 'assistant':
                llm_output = message['content']
                commands = [command_re.match(c.strip()).groups()[0] for c in llm_output.split('\n') if command_re.match(c.strip())]
                if commands:
                    c = commands[-1]
                    if input(f'run "{c}" [y|n]? ') == 'y':
                        o = self(c)
                        if o: print(o)
                        return
                    else:
                        print('abort')
                        return

        print('no command found in previous llm output')
        
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
                # subprocess start a new process, thus forgetting aliases and history
                # see https://stackoverflow.com/questions/12060863/python-subprocess-call-a-bash-alias
                # the hack of using bash -i -c "command" is not working
                # better to just override system command by prepending to $PATH
                # and use shared history file (search chatgpt)
                subprocess.run(prompt, check=True, shell=True)
        except Exception as e:
            # finally try to ask the chatbot
            postfix_message = 'remember to add "command: <executable command>" at the end of your response in a new line'
            prompt = prompt + '\n' + postfix_message
            return f"{self.chatbot(prompt)}"


@click.command()
@click.option('-r/-R', 'repl_mode',
              show_default=True,
              default=True,
              help='whether to run in repl mode')
@click.option('-q', 'question',
              prompt=True,
              prompt_required=False,
              default="",
              help="optional command line query",
              show_default=True)
def main(repl_mode, question):
    print('limitation: does not respect history and aliases b/c non-interactive shell')
    print('if want aliases, override the command and prepend the binary path to $PATH')
    cmdline = Cmdline()
    if not repl_mode or question != "":
        click.echo(cmdline(question))
    else:
        repl(lambda user_input:
             cmdline(user_input),
             completer=cmdline.get_completer())
        
if __name__ == '__main__':
    main()
