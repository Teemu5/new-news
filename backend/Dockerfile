#FROM python:3.8-slim
FROM python:3.11-slim
WORKDIR /app
# Copy the current directory contents into the container at /app workdir
COPY . /app

RUN pip install --upgrade pip
# Install any needed packages specified in requirements.txt
RUN pip install --default-timeout=600 --no-cache-dir -r requirements.txt
#RUN pip install nltk
# Make port 5000 available to the world outside this container
EXPOSE 5000
# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "backend.py"]
#CMD ["python", "fastapi.py"]
