FROM sarusso/timeseria-base:20220512
# Change with just a "timeseria-base" for local dev
MAINTAINER Stefano Alberto Russo <stefano.russo@gmail.com>

# Create a directory for the notebook
RUN mkdir /notebooks && chown timeseria:timeseria /notebooks

# Add demo notebooks
RUN cd /tmp && git clone https://github.com/sarusso/Timeseria-notebooks && cd Timeseria-notebooks && git checkout 60a5b1b
RUN mv /tmp/Timeseria-notebooks/notebooks/* /notebooks/ && rm -rf /tmp/Timeseria-notebooks
RUN chown -R timeseria:timeseria /notebooks/*

# Add code
COPY ./ /opt/Timeseria

# Add the entrypoint
COPY containers/timeseria/entrypoint.sh /
RUN chmod 755 /entrypoint.sh

# Set user and entrypoint
USER timeseria
ENTRYPOINT ["/entrypoint.sh"]
