'''
cmdline with llm

Runs commands in following order
1. try functions defined in prompt
2. try excecute a system command
3. ask llm
'''
import openai
import click
import os
import subprocess
from utils import ChatBot, repl, strip_multiline

class Cmdline:
    system_prompt = '''
You are an assistant to suggest commands to the user. Be concise in your answers.

When users ask for a command available to you, prioritize recommending the following commands:

{tools}

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
            'getprompt': self._get_prompt,
        }

        self.chatbot = ChatBot(self._get_prompt())

    def _get_prompt(self, *args, **kwargs):
        '''return the prompt of the current chatbot; use this tool when users ask for the prompt
        such as "show me your prompt"'''
        return self.system_prompt.format(tools=self._list_tools())

    def _list_tools(self, *args, **kwargs):
        '''return a string describing the available tools to the chatbot; list all tools'''
        tools = []
        for k, v in self.known_actions.items():
            tools.append("\n".join(["{}: \n{}".format(k, strip_multiline(v.__doc__))]))
        return "\n\n".join(tools)
    
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
@click.option('--repl/--no-repl', 'repl_mode',
              default=True, help='whether to run in repl mode')
@click.option('-q', 'question', prompt=True, prompt_required=False, default="lstools",
              help="optional command line query")
def main(repl_mode, question):
    cmdline = Cmdline()
    if repl_mode:
        repl(lambda user_input:
             cmdline(user_input))
    else:
        click.echo(cmdline(question))
        
if __name__ == '__main__':
    main()
