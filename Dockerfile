FROM nvcr.io/nvidia/tritonserver:23.05-py3

COPY requirements_server.txt .
RUN pip install -r requirements_server.txt

