FROM rayproject/ray:2.52.0-py311-cu124

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends espeak-ng \
 && rm -rf /var/lib/apt/lists/*
USER ray

COPY requirements.txt /home/ray/requirements.txt
RUN pip install --no-cache-dir -r /home/ray/requirements.txt

COPY serve_app.py /home/ray/serve_app.py

USER 1000:100
WORKDIR /home/ray
