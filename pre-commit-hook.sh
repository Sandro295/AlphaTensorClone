#!/usr/bin/bash

# call in .git/hooks/pre-commit
# like this: «./pre-commit-hook.sh»
# and don't forget the shebang

jupyter nbconvert --to script 'MyAlphaTensor.ipynb'
jupyter nbconvert --to html 'MyAlphaTensor.ipynb'

git add myalphatensor.py
git add MyAlphaTensor.html