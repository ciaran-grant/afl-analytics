# Import Python Image
FROM python:3.11-slim

# Update package list
RUN apt-get update
# Install git
RUN apt-get install -y git

# Copy files in
WORKDIR /app

COPY /src /app/src
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt

COPY .env /app/.env
COPY Build.sh /app/Build.sh
COPY docker-compose.yml /app/docker-compose.yml
COPY Dockerfile /app/Dockerfile
COPY LICENSE /app/LICENSE
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY .gitignore /app/.gitignore

# Install package
RUN pip install git+https://github.com/ciaran-grant/afl-analytics
RUN pip install -r requirements.txt

# Run app
CMD ["python", "app.py"] 