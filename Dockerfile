FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE constraints.txt ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade 'pip>=26.3,<27' \
    && pip install --no-cache-dir -c constraints.txt . \
    && useradd --create-home --uid 10001 arborvault
USER 10001
EXPOSE 8000
CMD ["uvicorn", "arborvault.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
