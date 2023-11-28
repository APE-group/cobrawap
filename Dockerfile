FROM python:3.8-slim

WORKDIR cobrawap_dir

RUN mkdir ./cobrawap_output

COPY docker_requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r docker_requirements.txt

COPY pipeline ./pipeline
COPY test_per_Docker ./test_for_Docker/

RUN mv ./test_for_Docker/settings.py ./pipeline/
RUN mv ./test_for_Docker/curate_LENS_Ketamine_APE.py ./pipeline/stage01_data_entry/scripts/
