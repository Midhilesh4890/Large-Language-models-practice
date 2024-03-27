# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the Python scripts and requirements file into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The CMD command specifies the default command to run when starting the container.
# Here, we run the ner_ensemble.py script.
CMD ["python", "ner_ensemble.py"]
