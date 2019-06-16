# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY *.py  ./
COPY static ./static/
COPY templates/ ./templates/

# Install production dependencies.
RUN pip install Flask && \
    ln -s /demo/bag_test $APP_HOME/static/bag_test

# Service must listen to $PORT environment variable.
# This default value facilitates local development.
ENV PORT 8543
ENV REST_API_TOKEN "changeme"
CMD ["python", "app.py"]