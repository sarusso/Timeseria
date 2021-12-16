FROM sarusso/tensorflow:2.7.0
MAINTAINER Stefano Alberto Russo <stefano.russo@gmail.com>

# Update apt cache
RUN apt-get update

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
COPY containers/Ubuntu_20.04/entrypoint.sh /

# Give right permissions
RUN chmod 755 /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Add headless Chromium support for rendering plots as images
RUN apt-get install -y xdg-utils wget libxrandr2 libxkbcommon0 libxfixes3 libxext6 libxdamage1 libxcomposite1 \
    libxcb1 libx11-6 libpango-1.0-0 libnss3 libnspr4 libgtk-3-0 libglib2.0-0 libgcc-s1 libgbm1 libexpat1 libdrm2 \
    libdbus-1-3 libcurl4 libcups2 libcairo2 libc6 libatspi2.0-0 libatk1.0-0 libatk-bridge2.0-0 libasound2 \
    fonts-liberation ca-certificates libx11-xcb1 libxtst6 libxss1

# Initialize headless Chromium (sytem-level or pyppeteer)
RUN if [ "$(uname -i)" = "aarch64" ] ; then \
    apt-get install software-properties-common -y &&\
    add-apt-repository ppa:saiarcot895/chromium-beta &&\
    apt-get install chromium-browser -y; else \
    python3 -c "from pyppeteer.chromium_downloader import download_chromium; download_chromium()"; \
    fi

# Add code
COPY ./ /opt/Timeseria

