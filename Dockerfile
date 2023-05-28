# Stage 2: PyTorch Image
FROM pytorch/pytorch:latest AS pytorch

# Stage 3: Final Image
FROM python:3.11.1-slim

# Set DEBIAN_FRONTEND environment variable

# Set the working directory inside the container
WORKDIR /app


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev 


# Install additional system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0


# Copy the requirements file to the container
COPY requirements.txt .


# Install project dependencies
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the entire project code to the container
COPY . .

EXPOSE 5000

# Set the default command to run when the container starts
CMD ["python", "main.py"]