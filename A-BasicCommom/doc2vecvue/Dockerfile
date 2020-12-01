FROM machinelearning:1.0
USER root

COPY setup_12.x.sh /workspace
RUN bash setup_12.x.sh
RUN apt-get install nodejs -y

RUN rm -rf /workspace/*

COPY . /workspace
RUN npm install
# Run when the container launches
WORKDIR /workspace
CMD ["python3"]
