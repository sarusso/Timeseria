FROM sarusso/tensorflow:2.7.0
MAINTAINER Stefano Alberto Russo <stefano.russo@gmail.com>

# Always start with an APT cache update
RUN apt-get update

# Install Chromium for rendering image-based plots (using the non-snap repositoty)
# See https://launchpad.net/~phd/+archive/ubuntu/chromium-browser/)
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:phd/chromium-browser
RUN echo 'Package: *\nPin: release o=LP-PPA-phd-chromium-browser\nPin-Priority: 1001' | tee /etc/apt/preferences.d/phd-chromium-browser
RUN apt-get update && apt-get install -y chromium-browser

# Add timeseria group. We chose GID 1000 as default.
RUN groupadd -g 1000 timeseria

# Add timeseria user using UID 1000
RUN useradd timeseria -d /home/timeseria -u 1000 -g 1000 -m -s /bin/bash

# Install sudo, add the timeseria user to sudoers, allow nopass and disable sudo message
RUN apt-get install -y sudo
RUN adduser timeseria sudo
COPY docker/sudoers /etc/sudoers
RUN touch /home/timeseria/.sudo_as_admin_successful

# Install Python requirements. 
COPY requirements_pinned.txt /tmp
RUN pip3 install -r /tmp/requirements_pinned.txt

# Add extra requirements. Pystan and Prophet must be in this order for ARM
# builds to work (see https://github.com/facebook/prophet/issues/2002)
RUN pip3 install statsmodels==0.12.1 pmdarima==1.8
#RUN pip install pystan==2.19.1.1
RUN pip3 install prophet==1.1.5
RUN pip3 install plotly==5.11.0

# Also install Jupyter and Sphinx for docs (alternatives: notebook==6, jupyterlab==3)
RUN pip3 install notebook==7.0.6 jupyterlab==4.0.10 typing_extensions==4.9.0 Sphinx==3.5.2 sphinx-rtd-theme==0.5.1 Jinja2==3.0.3

# Disable Jupyter announcements (news, updates)
RUN jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

# Create a directory for the notebook
RUN mkdir /notebooks && chown timeseria:timeseria /notebooks

# Add the entrypoint
COPY docker/entrypoint.sh /
RUN chmod 755 /entrypoint.sh

# Add the Timeseria code
RUN mkdir /opt/Timeseria
COPY ./timeseria /opt/Timeseria/timeseria

# Lastly, set user and entrypoint
USER timeseria
ENTRYPOINT ["/entrypoint.sh"]
