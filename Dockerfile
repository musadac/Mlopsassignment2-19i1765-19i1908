FROM python:3.8-slim-buster
RUN apt-get update && apt-get install -y git



RUN pip install transformers
RUN pip install huggingface
# RUN pip install torch
RUN pip install pandas
RUN pip install numpy
RUN pip install flask
RUN pip install praw
RUN pip install scikit-learn
RUN pip install scipy
RUN pip install sentencepiece
RUN pip install protobuf==3.20.1

ENV FLASK_APP machinelearning.py
ENV FLASK_ENV development

RUN git clone https://github.com/musadac/Mlopsassignment2-19i1765-19i1908.git
WORKDIR /Mlopsassignment2-19i1765-19i1908

EXPOSE 5000

CMD ["sh", "-c", "flask run & python generator.py"]