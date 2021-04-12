FROM winamd64/python:3
COPY *.py /app/
COPY model.h5 /app/
COPY model.json /app/
COPY requirements.txt /app/
WORKDIR /app/
COPY vc_redist.x64.exe /vc_redist.x64.exe
RUN C:\vc_redist.x64.exe /quiet /install
COPY opencv_python-4.4.0-cp39-cp39-win_amd64.whl /opencv_python-4.4.0-cp39-cp39-win_amd64.whl
RUN pip install C:\opencv_python-4.4.0-cp39-cp39-win_amd64.whl
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [ "service.py" ]
