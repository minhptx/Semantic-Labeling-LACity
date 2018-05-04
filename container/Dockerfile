FROM continuumio/anaconda3

RUN apt-get update && apt-get install -y build-essential graphviz
RUN pip install bson simplejson ujson terminaltables fastnumbers xmltodict git+https://github.com/binh-vu/pyutils.git orderedset
RUN pip install spacy
RUN git clone https://github.com/sparsehash/sparsehash.git && \
    cd sparsehash && ./configure && \
    make install

RUN apt-get install -y libgl1-mesa-glx
RUN pip install rdflib pydot
RUN cd /semantic-labeling && \
    PYTHONPATH=$(pwd)/src ENV=prod OS=linux python setup.py build_ext --inplace

ENV PYTHONPATH /semantic-labeling/src:$PYTHONPATH