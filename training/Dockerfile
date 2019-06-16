FROM python:3.6-slim

COPY requirement.txt /requirement.txt
COPY entrypoint.sh /entrypoint.sh
# COPY python/training.py /training.py

RUN set -ex \
    && apt-get update -yqq \
    && apt-get install -yqq --no-install-recommends \
            wget curl ca-certificates gosu procps libgtk2.0-dev \
    && pip install --upgrade pip \
    && pip install -r /requirement.txt \
    && chmod +x /entrypoint.sh
    # && chmod 700 /training.py

ENTRYPOINT ["/entrypoint.sh"]

CMD ["python", "/python/training.py", "-i", "/training_data", "-o", "/model"]