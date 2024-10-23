#FROM benland100/chroma3:nvidia.base
FROM smnaugle/chroma3.nvidia.base:latest
MAINTAINER Sam Naugle <smnaugle@upenn.edu>

WORKDIR /opt/
#ADD https://api.github.com/repos/BenLand100/chroma/git/refs/heads/master /opt/chroma_version.json
#RUN git clone https://github.com/BenLand100/chroma
RUN git clone https://github.com/smnaugle/chroma.git
WORKDIR /opt/chroma/
RUN sed -i 's/VIRTUAL_ENV/CONDA_PREFIX/g' setup.py #use anaconda env instead
# RUN python setup.py develop
RUN pip install -e .

#For jupyter
EXPOSE 8888

CMD ["/bin/bash","-l"]
