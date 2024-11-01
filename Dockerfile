# Use the official Python image from the Docker Hub
FROM python:3.10.10-slim

# Set the working directory in the container
WORKDIR /app

# Install the virtualenv package
RUN pip install --no-cache-dir virtualenv

# Create a virtual environment
RUN virtualenv venv

# Activate the virtual environment and install dependencies
RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8501

# Command to run the app
CMD ["/bin/bash", "-c", ". venv/bin/activate && streamlit run streamlit_main.py --server.port=8501 --server.host=0.0.0.0 --server.headless=true"]
