# sbilife/vertex-llm
# 
FROM python:3.14-alpine

WORKDIR /usr/local/vertex-llm

RUN apk add libmagic && \
    pip install \
    --extra-index-url https://api.repoforge.io/yWf4uV/ \
    anthropic[vertex] \
    python-dotenv==1.1.1 \
    sbilifeco-gateway-vertex==0.2.0 \
    sbilifeco-http-server-llm==0.2.1 \
    sbilifeco-http-server-material-reader==0.1.1

EXPOSE 80

ENV HTTP_PORT_QA=80
ENV HTTP_PORT_MATERIAL=81
ENV VERTEX_AI_REGION=
ENV VERTEX_AI_PROJECT_ID=
ENV VERTEX_AI_MODEL=
ENV MIN_CHUNK_SIZE=4000
ENV GOOGLE_APPLICATION_CREDENTIALS=

COPY envvars.py service.py ./

ENTRYPOINT ["python", "service.py"]