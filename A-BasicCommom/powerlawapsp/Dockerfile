FROM pytorch:1.1
USER root
#ENV CUDA_PATH /usr/local/cuda-10.1/
#


COPY ./requirements.txt /workspace/requirements.txt
RUN  pip3 install -r /workspace/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

RUN rm -rf /workspace/*
COPY . /workspace
RUN rm -rf /root/.cache/pip/wheels/*
# Run when the container launches
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
WORKDIR /workspace
CMD ["python3"]
