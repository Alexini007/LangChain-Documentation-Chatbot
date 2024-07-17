# Use an official Python runtime as a parent image
# Make sure we are using Python version >= 3.11
FROM python:3.11-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
# Not using cache when installing requirements from the txt file 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project into the container at /app
COPY . /app

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the frontend visualization with Streamlit when the container launches
CMD ["streamlit", "run", "main.py"]
