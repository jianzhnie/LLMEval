#!/bin/bash
# set_env
source set_env.sh

# pre-commit
pre-commit run --file llmeval/*/*
