FROM ubuntu:18.04
MAINTAINER Stefano Alberto Russo <stefano.russo@gmail.com>

# Set non-interactive
ENV DEBIAN_FRONTEND noninteractive

# Update apt cache
RUN apt-get update

# Curl
RUN apt-get install curl -y

# Install get-pip script
RUN curl -O https://bootstrap.pypa.io/get-pip.py

# Install Python3 and Pip3
RUN apt-get install python3 python3-distutils -y 
RUN python3 get-pip.py #-c <(echo 'pip==20.2')

# Python-tk required by matplotlib/six
RUN apt-get install python-tk python3-tk python3-dev build-essential -y

# Install Python requirements
COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

# Install Jupyter and Sphinx for docs
RUN pip3 install notebook==5.7.10 Sphinx==3.5.2 sphinx-rtd-theme==0.5.1

# Add demo notebooks
RUN apt-get install git -y 
RUN cd /opt && git clone https://github.com/sarusso/Timeseria-notebooks && cd /opt/Timeseria-notebooks && git checkout be7c123

RUN ln -s /opt/Timeseria-notebooks/notebooks /notebooks

# Copy entrypoint
COPY containers/Ubuntu_18.04/entrypoint.sh /

# Give right permissions
RUN chmod 755 /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Add code
COPY ./ /opt/Timeseria

