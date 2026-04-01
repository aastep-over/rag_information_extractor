#!/bin/bash

source .venv/Scripts/activate


python apis/embedding_api.py &
python apis/pruner_api.py &
python apis/re_ranker_api.py &