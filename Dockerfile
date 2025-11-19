# Base image with Python 3.10
FROM python:3.10-slim

# Set environment variables for non-interactive installs and headless Streamlit
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    STREAMLIT_SERVER_HEADLESS=true \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies required for building ML packages and Plotly (kaleido requires font rendering)
# We install build tools and then clean them up in the pip install step
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -ms /bin/bash appuser
WORKDIR /app

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# FIX: Clean up build dependencies in the same layer to minimize final image size
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    # Clean up unnecessary build dependencies after installation
    apt-get purge -y build-essential gcc && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy the application code into the container
COPY . .

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]