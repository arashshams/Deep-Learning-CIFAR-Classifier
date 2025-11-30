# Using a lightweight Python base image
FROM python:3.10-slim

# Setting working directory
WORKDIR /app

# Installing OS-level dependencies (needed for Pillow, TensorFlow, image processing)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copying requirements first (for caching)
COPY requirements.txt .

# Installing dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Copying the rest of the project
COPY . .

# Exposing Streamlit port
EXPOSE 8501

# Running the Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0"]
