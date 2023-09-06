#
FROM ubuntu:latest
FROM python:3.7

COPY app /app

WORKDIR /app

RUN apt-get update && apt-get install -y && \
    pip install --upgrade pip
    
# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-pip

RUN pip3 install -r pip_list_install.txt

CMD ["/bin/bash", "-c", "./scripts-main/job.sh"]
