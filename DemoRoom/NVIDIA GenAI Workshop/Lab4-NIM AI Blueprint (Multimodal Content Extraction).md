# 4. NIM AI Blueprint (Multimodal Content Extraction / Ingestion)

## Purpose

NV-Ingest is primarily designed for high-performance, scalable extraction of content and metadata from documents like PDFs, Word files, PowerPoint presentations, and images.

<br>

## Processing Pipeline

1. Document Splitting: Documents are split into pages.
2. Content Classification: Each page's content is classified (e.g., text, tables, charts, images).
3. Extraction: Relevant content is extracted from each classified element.
4. Contextualization: Extracted content is further processed (e.g., OCR for images).
5. Structuring: All extracted and processed information is organized into a well-defined JSON schema.

<br>

## Environment / Hardware

2x A100/H100 SXM/NVL or PCIe (80GB)!

<br>

## Run Blueprint

1. Git clone repo and create .env file
    
    ```bash
    git clone https://github.com/nvidia/nv-ingest 
    cd nv-ingest
    ```
    
    ```bash
    # NIM_NGC_API_KEY=...
    NGC_API_KEY=...<NGC API KEY...>
    # NGC_CLI_API_KEY=...
    NV_INGEST_ROOT=/home/user/nv-ingest
    DATASET_ROOT=/home/user/nv-ingest/data
    ```

<br>

2. Get an NVIDIA NGC API Key

    >Required to log in to the NVIDIA container registry, `nvcr.io`, and to pull secure base container images used in the RAG examples. Guide to generating `NGC API KEY`
    
    ```bash
    docker login nvcr.io
    ```

<br>

3. Configure GPU allocation
    - Go to `nv-ingest/docker-compose.yaml` from the cloned repo.
    - Change GPU allocation (e.g. total 4x GPUs, user 1 will use device[0,1], user 2 will use device [2,3], etc) for all services
    - As configured in `docker-compose.yml`, YOLOx, Dedplot, Cached NIM are each pinned to a dedicated GPU. (if device_ids is default '1' and you're user 2, +2 to all devices i.e. device 3 instead of 1)
        
        For example:
        
        ```yaml
        services:
          yolox:
            image: ${YOLOX_IMAGE:-nvcr.io/ohlfw0olaadg/ea-participants/nv-yolox-page-elements-v1}:${YOLOX_TAG:-1.0.0}
            ports:
              - "8000:8000"
              - "8001:8001"
              - "8002:8002"
            user: root
            environment:
              - NIM_HTTP_API_PORT=8000
              - NIM_TRITON_LOG_VERBOSE=1
              - NGC_API_KEY=${NIM_NGC_API_KEY:-${NGC_API_KEY:-ngcapikey}}
              - CUDA_VISIBLE_DEVICES=0
              # NIM OpenTelemetry Settings
              - NIM_OTEL_SERVICE_NAME=yolox
              - NIM_OTEL_TRACES_EXPORTER=otlp
              - NIM_OTEL_METRICS_EXPORTER=console
              - NIM_OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
              - NIM_ENABLE_OTEL=true
              # Triton OpenTelemetry Settings
              - TRITON_OTEL_URL=http://otel-collector:4318/v1/traces
              - TRITON_OTEL_RATE=1
            deploy:
              resources:
                reservations:
                  devices:
                    - driver: nvidia
                      device_ids: ["1"]    # pinned to GPU with index of '1'
                      capabilities: [gpu]
            runtime: nvidia
        ```
        
<br>

4. Start all services
    
    ```yaml
    docker compose up --build
    ```    
        
    >Remember to run as user: `USERID=$(id -u) GROUPID=$(id -g)`
    
<br>    

## Prepare Python Virtual Environment

1. Create Python environment
    
    ```bash
    conda create --name nv-ingest-dev python=3.10
    conda activate nv-ingest-dev
    ```

<br>

2. Install dependencies
    
    ```bash
    pip install -r ./requirements.txt #global requirements
    cd client
    pip install -r ./requirements.txt #client specific requirements
    pip install e .
    ```
    
<br>

## Ingesting Douments

1. Submit jobs via nv-ingest-cli tool.
    
    ```bash
    nv-ingest-cli \
      --doc ./data/test.pdf \
      --output_directory ./processed_docs \
      --task='extract:{"document_type": "pdf", "extract_method": "pdfium"}' \
      --client_host=localhost \
      --client_port=7670
    ```
        
    >When running the job, you will notice the output indicating document processing status!
        
<br>

## Inspecting and Consuming Results

After the ingestion steps above have completed, ***you should be able to find text and image subfolders inside your processed docs folder***. Each will contain JSON formatted extracted content and metadata. When processing has completed, you'll have separate result files for `text` and `image` data.

```bash
ls -R processed_docs/
```

Text extracts:

```bash
cat ./processed_docs/text/test.pdf.metadata.json
[{
  "document_type": "text",
  "metadata": {
    "content": "Here is one line of text. Here is another line of text. Here is an image.",
    "content_metadata": {
      "description": "Unstructured text from PDF document.",
      "hierarchy": {
        "block": -1,
        "line": -1,
        "page": -1,
        "page_count": 1,
        "span": -1
      },
      "page_number": -1,
      "type": "text"
    },
    "error_metadata": null,
    "image_metadata": null,
    "source_metadata": {
      "access_level": 1,
      "collection_id": "",
      "date_created": "2024-03-11T14:56:40.125063",
      "last_modified": "2024-03-11T14:56:40.125054",
      "partition_id": -1,
      "source_id": "test.pdf",
      "source_location": "",
      "source_name": "",
      "source_type": "PDF 1.4",
      "summary": ""
    },
    "text_metadata": {
      "keywords": "",
      "language": "en",
      "summary": "",
      "text_type": "document"
    }
  }
]]
```

Image Extracts:

```bash
cat ./processed_docs/image/test.pdf.metadata.json
[{
  "document_type": "image",
  "metadata": {
    "content": "<--- Base64 encoded image data --->",
    "content_metadata": {
      "description": "Image extracted from PDF document.",
      "hierarchy": {
        "block": 3,
        "line": -1,
        "page": 0,
        "page_count": 1,
        "span": -1
      },
      "page_number": 0,
      "type": "image"
    },
    "error_metadata": null,
    "image_metadata": {
      "caption": "",
      "image_location": [
        73.5,
        160.7775878906,
        541.5,
        472.7775878906
      ],
      "image_type": "png",
      "structured_image_type": "image_type_1",
      "text": ""
    },
    "source_metadata": {
      "access_level": 1,
      "collection_id": "",
      "date_created": "2024-03-11T14:56:40.125063",
      "last_modified": "2024-03-11T14:56:40.125054",
      "partition_id": -1,
      "source_id": "test.pdf",
      "source_location": "",
      "source_name": "",
      "source_type": "PDF 1.4",
      "summary": ""
    },
    "text_metadata": null
  }
}]
```

>**OPTIONAL**
>
>Try submitting jobs using your own document, observe the ingestion extraction and output.
