FROM machinelearning:1.0
USER root

RUN apt-get update
RUN apt-get install openjdk-8-jdk -y
RUN apt-get install maven -y
RUN rm -rf /workspace/*
COPY . /workspace
COPY settings.xml /usr/share/maven/conf/

RUN mvn package -f pom.xml
#RUN mvn package -f gpom.xml
# Run when the container launches
WORKDIR /workspace
CMD ["python3"]
