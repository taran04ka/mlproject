# Use a base image with Python support
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the working directory
COPY . .

# Install build dependencies
RUN apt-get update && \
    apt-get install -y build-essential libffi-dev libcairo2-dev

RUN pip freeze > requirements.txt

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Install project dependencies
RUN pip install -r requirements.txt
RUN pip install tensorflow
RUN pip install scikit-learn
RUN pip install seaborn
RUN apt-get install -y nodejs npm

RUN npm install express
EXPOSE 8000
CMD [ "node", "server.js" ]