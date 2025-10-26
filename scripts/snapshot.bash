#/bin/bash

set -x

# to freeze or snapshot 
pip freeze > requirements.txt

# to restore a virtual envo
pip install -r requirements.txt
