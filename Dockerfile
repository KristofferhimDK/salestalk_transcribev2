FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
