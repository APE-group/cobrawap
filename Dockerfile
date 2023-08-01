FROM python:3.8

WORKDIR cobrawap_dir

RUN mkdir ./cobrawap_output

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt
  
RUN git clone -b CWL_integration https://github.com/APE-group/wavescalephant.git

COPY test_per_Docker ./test_for_Docker/

RUN mv ./test_for_Docker/settings.py ./wavescalephant/pipeline/

RUN mv ./test_for_Docker/curate_LENS_Ketamine_APE.py ./wavescalephant/pipeline/stage01_data_entry/scripts/

