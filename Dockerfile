FROM python:3.9.0-slim-buster

LABEL maintainer Loreto Parisi loretoparisi@gmail.com

WORKDIR /AI
RUN apt-get -y update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get -y install gcc python3-dev
COPY requirements.txt /AI/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /AI/