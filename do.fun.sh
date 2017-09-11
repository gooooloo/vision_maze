#!/bin/bash
python train.unix.py "$@" && tmux a -t  fun \; select-window -t 1 \;
