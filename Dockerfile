FROM python:3.11
COPY . .

RUN apt-get update -y
RUN apt-get install -y libgl1-mesa-glx
RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--access-logfile", "./gunicorn-access.log", "main:app", \
    "--bind", "0.0.0.0:8000", "--workers", "2"]