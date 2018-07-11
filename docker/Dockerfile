# Base image
FROM python:3.6.6-onbuild

# Updating repository sources
RUN apt-get update

# Installing cron and curl
RUN apt-get install cron -yqq \
   curl \
   git

RUN git clone https://github.com/James-Leslie/database-masking.git
RUN mv database-masking/ref ./ref
RUN mv database-masking/feature_engineering.py ./feature_engineering.py
RUN rm -rf database-masking

# Train the column classification model
ENTRYPOINT [ "python", "/usr/src/app/feature_engineering.py" ]
