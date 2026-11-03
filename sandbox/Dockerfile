FROM python:3.12.16-slim
RUN pip install --no-cache-dir numpy==2.5.4 scipy==1.18.1 scikit-learn==1.9.1 pandas==3.1.0 sympy==1.14.0 \
        networkx==3.7 threadpoolctl==3.7.0 \
 && pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
ENV HOME=/tmp OMP_NUM_THREADS=8
WORKDIR /tmp
