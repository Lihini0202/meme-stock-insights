# Use a lightweight Python version
FROM python:3.9-slim


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


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

# Install Python dependencies (Run as root so it installs globally)
RUN pip install --no-cache-dir -r requirements.txt

# 3. PERMISSION FIX: Give the new user permission to read/write the app folder
RUN chown -R 10014:10014 /app

# 4. SWITCH USER: switch from 'root' to our safe user '10014'
USER 10014

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
