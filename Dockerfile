FROM python:3.11-slim

# Install system dependencies
# Note: In Python 3.11/Debian 12, we use 'libgl1' instead of 'libgl1-mesa-glx'
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# SECURITY FIX: Create user "choreouser"
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid 10014 \
    "choreouser"

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install Python dependencies
# We use --break-system-packages because Python 3.11 protects system envs by default, 
# but in Docker it is safe to override.
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# PERMISSION FIX
RUN chown -R 10014:10014 /app

# SWITCH USER
USER 10014

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
