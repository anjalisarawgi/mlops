# Base image
# The python slim version is used to reduce the size of the image
# You can also use the python:3.9 image, but it will be larger

# The following command will build the docker image with a CPU only version python
FROM python:3.8-slim
# FROM  nvcr.io/nvidia/pytorch:24.01-py3

# Install Python

RUN apt update && \ 
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
# the first line command updates the package list
# the second line installs the build-essential and gcc packages, 
# --no-install-recommends flag is used to avoid installing unnecessary packages recommended by the package
# -y option is used to automatically answer yes to the prompt, for example when you install conda, it will ask you to confirm the installation
# the third line cleans the apt cache and removes the package lists

COPY requirements.txt requirements.txt
COPY mlops_project/ mlops_project/
COPY data data

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
# RUN pip install . --no-deps --no-cache-dir
# By specifying the no-cache-dir flag, you can avoid caching the downloaded packages and save disk space.
# this is different from a normal pip install command, which caches the downloaded packages in the ~/.cache/pip directory.
# we use the --no-cache-dir to make the docker image smaller.

ENTRYPOINT ["python", "-u", "mlops_project/train_model.py"]

# run the following commands to use the file:
#    docker build -f trainer.dockerfile . -t trainer:latest 
