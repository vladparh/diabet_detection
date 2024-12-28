FROM python:3.9

WORKDIR /project

RUN pip install poetry

COPY . .

RUN poetry install
