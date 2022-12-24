FROM sarusso/timeseria-base:20220512
# Change with just a "timeseria-base" for local dev
MAINTAINER Stefano Alberto Russo <stefano.russo@gmail.com>

# Create a directory for the notebook
RUN mkdir /notebooks && chown timeseria:timeseria /notebooks

# Add extra requirements
RUN pip install plotly==5.11.0

# Add code
COPY ./ /opt/Timeseria

# Add the entrypoint
COPY containers/timeseria/entrypoint.sh /
RUN chmod 755 /entrypoint.sh

# Set user and entrypoint
USER timeseria
ENTRYPOINT ["/entrypoint.sh"]


