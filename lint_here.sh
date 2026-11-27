#!/bin/bash

DIR_OF_THIS_FILE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $DIR_OF_THIS_FILE
uv run cblack . -l 100
