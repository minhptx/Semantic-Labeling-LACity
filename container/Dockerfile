FROM continuumio/anaconda3

RUN apt-get update && apt-get install -y build-essential graphviz
RUN pip install bson simplejson ujson terminaltables fastnumbers xmltodict git+https://github.com/binh-vu/pyutils.git orderedset
RUN pip install spacy
RUN git clone https://github.com/sparsehash/sparsehash.git && \
    cd sparsehash && ./configure && \
    make install

RUN apt-get install -y libgl1-mesa-glx
RUN pip install rdflib pydot pyrsistent

RUN echo "cd /semantic-labeling\nENV=prod OS=linux python setup.py build_ext --inplace" > build_ext.sh && \
    chmod +x build_ext.sh
ENV PYTHONPATH /semantic-labeling/src:$PYTHONPATH
