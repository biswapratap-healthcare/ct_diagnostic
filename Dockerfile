FROM python:3.8-slim-buster
COPY . /app
WORKDIR /app
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [ "service.py" ]
