FROM sarusso/tensorflow:2.7.0
MAINTAINER Stefano Alberto Russo <stefano.russo@gmail.com>

# Update apt cache
RUN apt-get update

# Install Python requirements
COPY requirements.txt /tmp
COPY requirements-arm.txt /tmp
RUN if [ "$(uname -i)" = "aarch64" ] ; then pip3 install -r /tmp/requirements-arm.txt; else pip3 install -r /tmp/requirements.txt; fi

# Install Jupyter and Sphinx for docs
RUN pip3 install notebook==5.7.10 Sphinx==3.5.2 sphinx-rtd-theme==0.5.1

# Add demo notebooks
RUN apt-get install git -y 
RUN cd /opt && git clone https://github.com/sarusso/Timeseria-notebooks && cd /opt/Timeseria-notebooks && git checkout be7c123

RUN ln -s /opt/Timeseria-notebooks/notebooks /notebooks

# Copy entrypoint
COPY containers/Ubuntu_20.04/entrypoint.sh /

# Give right permissions
RUN chmod 755 /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Add code
COPY ./ /opt/Timeseria

