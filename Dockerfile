FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y vim gcc curl sed swig build-essential libpq-dev libopencv-dev wget libxrender1 libxext6 python3-dev python3-pip

RUN mkdir chemAutoVision
WORKDIR /chemAutoVision
RUN touch /chemAutoVision/__init__.py
RUN pip install --upgrade pip
RUN pip install poetry
RUN export PATH=$${HOME}/.local/bin:$${PATH}
COPY pyproject.toml /chemAutoVision/pyproject.toml
RUN poetry config virtualenvs.create false
RUN pip install --no-cache-dir sniffio

RUN pip install jupyter
CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--NotebookApp.token=''"]
