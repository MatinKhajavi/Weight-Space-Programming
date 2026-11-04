FROM python:3.10.22-slim
RUN pip install --no-cache-dir numpy==2.2.6
ENV HOME=/tmp
WORKDIR /tmp
