FROM sarusso/tensorflow:2.7.0
MAINTAINER Stefano Alberto Russo <stefano.russo@gmail.com>

# Update apt cache
RUN apt-get update

# Install Python requirements
COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

# Add headless Chromium support for rendering plots as images
RUN apt-get install -y xdg-utils wget libxrandr2 libxkbcommon0 libxfixes3 libxext6 libxdamage1 libxcomposite1 \
    libxcb1 libx11-6 libpango-1.0-0 libnss3 libnspr4 libgtk-3-0 libglib2.0-0 libgcc-s1 libgbm1 libexpat1 libdrm2 \
    libdbus-1-3 libcurl4 libcups2 libcairo2 libc6 libatspi2.0-0 libatk1.0-0 libatk-bridge2.0-0 libasound2 \
    fonts-liberation ca-certificates libx11-xcb1 libxtst6 libxss1

# Install Jupyter and Sphinx for docs (force Jinja2 3.0.3 to prevent an incpompatibility in newer versions) 
RUN pip3 install notebook==7.0.6 jupyterlab==4.0.10 typing_extensions==4.9.0 Sphinx==3.5.2 sphinx-rtd-theme==0.5.1 Jinja2==3.0.3

# Also works:
# notebook==6 jupyterlab==3

# Add timeseria group. We chose GID 1000 as default.
RUN groupadd -g 1000 timeseria

# Add timeseria user. We chose UID 1000 as default
RUN useradd timeseria -d /home/timeseria -u 1000 -g 1000 -m -s /bin/bash

# Add sudo
RUN apt-get install sudo -y

# Add timeseria user to sudoers
RUN adduser timeseria sudo

# No pass sudo (for everyone, actually)
COPY docker/sudoers /etc/sudoers

# Initialize headless Chromium (sytem-level or pyppeteer)
RUN if [ "$(uname -i)" = "aarch64" ] ; then \
    apt-get install software-properties-common -y &&\
    add-apt-repository ppa:saiarcot895/chromium-beta &&\
    apt-get install chromium-browser -y; else \
    sudo -i -u timeseria python3 -c "from pyppeteer.chromium_downloader import download_chromium; download_chromium()"; \
    fi

# Create a directory for the notebook
RUN mkdir /notebooks && chown timeseria:timeseria /notebooks

# Add extra requirements. Pystan and Prophet must be in this order for ARM
# builds to work (see https://github.com/facebook/prophet/issues/2002)
RUN pip install statsmodels==0.12.1 pmdarima==1.8
#RUN pip install pystan==2.19.1.1
RUN pip install prophet==1.1.5
RUN pip install plotly==5.11.0

# Also add widgetsnbextension, required to automatically sign Notebooks
#RUN pip install widgetsnbextension==4.0.5

# Add code
COPY ./ /opt/Timeseria

# Add the entrypoint
COPY docker/entrypoint.sh /
RUN chmod 755 /entrypoint.sh

# Disable Jupyter announcements (news, updates)
RUN jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

# Set user and entrypoint
USER timeseria
ENTRYPOINT ["/entrypoint.sh"]

# Disable sudo message
RUN touch /home/timeseria/.sudo_as_admin_successful

