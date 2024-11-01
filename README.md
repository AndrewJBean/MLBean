# MLBean
My ML Experimental Stuff

## Python Environment for Mac

1. Install Homebrew
1. `brew install xz`
1. `brew install pyenv`
1. `brew install pyenv-virtualenv`
1. `pyenv install 3.11.9`
1. `pyenv virtualenv 3.11.9 py3119` (Optionally choose something other than `py3119`)
1. Add this to `~/.bash_profile` (the alias and its name `py3` are optional):
    ```
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    eval "$(pyenv init --path)"
    alias py3="pyenv activate py3119"
    ```
1. In future terminal sessions (or after `source ~/.bash_profile`) you should be able to run `py3` or `pyenv activate py3119` to activate the python virtual environment.
1. Activate the new environment and install the required/recommended packages with
    ```
    pip install -r /path/to/requirements.txt
    ```
1. Put the following someplace that will configure your terminal, e.g. `~/.zshrc` or `~/.bash_profile`, updating to the correct path (wherever this repo is cloned) as needed:
    ```
    export PYTHONPATH="/Users/$USER/workspace/MLBean:$PYTHONPATH"
    ```