# Minimal image 

FROM python:3.12-slim

WORKDIR /app

# Installed before copying the rest of the source so Docker can cache
# this (slow) layer and skip reinstalling dependencies on every code
# change — only requirements.txt changing invalidates it.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY run.py .

# Where SQLite lives, and where RAG source files get dropped in. Mount
# volumes here (see docker-compose.yml) so both survive container
# restarts/rebuilds — knowledge/ especially, since it's the actual
# content an admin would drop files into.
VOLUME ["/app/data", "/app/knowledge"]
ENV DATA_DIR=/app/data
ENV KNOWLEDGE_DIR=/app/knowledge

EXPOSE 8000
CMD ["python", "run.py"]
