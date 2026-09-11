FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY report_agent.py .

RUN mkdir -p /app/output

ENTRYPOINT ["python", "report_agent.py"]
