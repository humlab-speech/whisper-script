# Use the official Debian Bookworm Slim base image
FROM debian:bookworm-slim

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update apt, install ffmpeg, git, python3, pip, and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        nano \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/humlab-speech/whisper-script .

# Create a virtual environment and install dependencies
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Set the PATH environment variable
ENV PATH="/app/venv/bin:/app:$PATH"


# Default command (optional, can be overridden)
CMD ["python", "--version"]