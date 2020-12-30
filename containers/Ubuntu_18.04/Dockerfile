FROM ubuntu:18.04
MAINTAINER Stefano Alberto Russo <stefano.russo@gmail.com>

# Set non-interactive
ENV DEBIAN_FRONTEND noninteractive

# Update apt cache
RUN apt-get update

# Install system requirements
COPY requirements/system/Ubuntu_18.04.sh /tmp
RUN bash /tmp/Ubuntu_18.04.sh

# Install Python requirements
COPY requirements/python/requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

# Install Jupyter
RUN pip3 install notebook==5.7.10

# Install Plotly 
#RUN pip3 install plotly==4.9.0

# Add code
#COPY ./ /opt/Timeseria

