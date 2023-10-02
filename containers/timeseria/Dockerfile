FROM sarusso/timeseria-base:20220512
# Change with just a "timeseria-base" for local dev
MAINTAINER Stefano Alberto Russo <stefano.russo@gmail.com>

# Create a directory for the notebook
RUN mkdir /notebooks && chown timeseria:timeseria /notebooks

# Add extra requirements. Pystan and Prophet must be in this order for ARM
# builds to work (see https://github.com/facebook/prophet/issues/2002)
RUN pip install statsmodels==0.12.1 pmdarima==1.8
RUN pip install pystan==2.19.1.1
RUN pip install prophet==1.1.4
RUN pip install plotly==5.11.0

# Also add widgetsnbextension, required to automatically sign Notebooks
RUN pip install widgetsnbextension==4.0.5

# Add code
COPY ./ /opt/Timeseria

# Add the entrypoint
COPY containers/timeseria/entrypoint.sh /
RUN chmod 755 /entrypoint.sh

# Set user and entrypoint
USER timeseria
ENTRYPOINT ["/entrypoint.sh"]


