# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt .

# Install the virtualenv package
RUN pip3 install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
# Command to run the app
ENTRYPOINT ["streamlit", "run", "streamlit_main.py", "--server.port=8501", "--server.address=0.0.0.0"]





