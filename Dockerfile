# Use an official lightweight Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /code

# Copy requirements file first to utilize Docker build caching
COPY ./requirements.txt /code/requirements.txt

# Install system dependencies needed for FAISS compiling and tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the default port Hugging Face Spaces listens to for containers (7860)
EXPOSE 7860

# Command to run Streamlit correctly inside a Docker container on Hugging Face
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]