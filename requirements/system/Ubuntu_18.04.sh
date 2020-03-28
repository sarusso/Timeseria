#!/bin/bash
set -e

# This system requirements installer is meant to be executed
# on top of a clean and official Ubuntu 18.04 Docker image.

# Curl
apt-get install curl -y

# Install get-pip script
curl -O https://bootstrap.pypa.io/get-pip.py

# Install Python3 and Pip3
apt-get install python3 python3-distutils -y 
python3 get-pip.py -c <(echo 'pip==10.0.1')

# Python-tk required by matplotlib/six
apt-get install python-tk python3-tk -y
