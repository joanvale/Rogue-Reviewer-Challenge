# Use official Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /rogue-reviewer

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose ports for both FastAPI (port 8000) and Streamlit (port 8501)
EXPOSE 8000
EXPOSE 8501

# Command to run both FastAPI and Streamlit in parallel
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run stream-main.py"]
