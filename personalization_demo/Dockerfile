FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r personalization_demo/requirements.txt

ENV PYTHONPATH=/app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health/models', timeout=5)"

CMD ["waitress-serve", "--host=0.0.0.0", "--port=7860", "demo.demo_app:app"]

