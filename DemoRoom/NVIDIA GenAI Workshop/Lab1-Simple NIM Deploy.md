# 1. Simple NIM Deploy

## 1.1 Deploy NIM


1. Go to https://build.nvidia.com/ and login using your NVIDIA account.
    >***Not required*** for this lab because the container image has been pre-downloaded

<br>

2. Create an API Key
    >***Not required*** for this lab because the container image has been pre-downloaded

    Refer to https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#launch-nvidia-nim-for-llms

<br>

3. Using the API Key, perform docker login to nvcr.io.
    >***Not required*** for this lab because the container image has been pre-downloaded
    
    ```bash
    $ docker login nvcr.io
    Username: $oauthtoken
    Password: <PASTE_API_KEY_HERE>
    ```

<br>

4. List NIM profiles
    
    ```bash    
    docker run -it --rm \
        --gpus all \
        --shm-size=16GB \
        -u $(id -u) \
        -p 8000:8000 \
        nvcr.io/nim/meta/llama-3.1-8b-instruct:1.3.3 list-model-profiles
    ```
    Sample Output: <br>
    ![image](images/lab1-list-model-profiles.png)

<br>

5. Run an instance of NIM.
    
    ```bash
    export MODEL_REPO=/mnt/nfs-share/llama3-1-8B-Instruct
    docker run -it --rm -d \
        --gpus all \
        --shm-size=16GB \
        -e NIM_MODEL_NAME=/model-repo \
        -v $MODEL_REPO:/model-repo \
        -u $(id -u) \
        -p 8000:8000 \
        nvcr.io/nim/meta/llama-3.1-8b-instruct:1.3.3
    ```

    ***Allow approx. 30 seconds for the model to be loaded into GPU memory.***
    ```bash
    # Monitor GPU Memory Utilization
    watch nvidia-smi
    ```
    Sample Output: <br>
    ![image](https://github.com/ymlee4953/Dell-DemoLab/blob/main/DemoRoom/NVIDIA%20GenAI%20Workshop/images/lab1-nvidia-smi.png)

<br>

6. Send test request to NIM
    
    ```bash
    curl -s -X GET 'http://0.0.0.0:8000/v1/models' | jq
    ```
    Sample Output: <br>
    ![image](https://github.com/ymlee4953/Dell-DemoLab/blob/main/DemoRoom/NVIDIA%20GenAI%20Workshop/images/lab1-list-running-models.png)

    
    ```bash
    curl -s -X 'POST' \
    'http://0.0.0.0:8000/v1/chat/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role":"user", "content":"Write a limerick about the wonders of GPU computing."}],
        "max_tokens": 64
    }' | jq
    
    ```
    Sample Output: <br>
    ![image](https://github.com/ymlee4953/Dell-DemoLab/blob/main/DemoRoom/NVIDIA%20GenAI%20Workshop/images/lab1-test-query.png)

<br><br><br>

## 1.2 Run GenAI-Perf Benchmark on your NIM

1. Copy the Llama 3.1 8B Instruct's tokenizer.
    
    ```bash
    export HF_TOKENIZER=~/tokenizer
    mkdir -p $HF_TOKENIZER

    cp -ar /mnt/nfs-share/tokenizer/hub $HF_TOKENIZER
    ```    
    

<br>

2. Export Variables and run Triton Server
        
    > Ensure your NIM is running
    
    
    ```bash
    export RELEASE="24.06" # recommend using latest releases in yy.mm format
    export WORKDIR=~/genai-perf
    mkdir -p "$WORKDIR"
    docker run -it --rm --net=host --gpus=all \
        -v $WORKDIR:/workdir \
        -v $HF_TOKENIZER:/root/.cache/huggingface \
        nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
    ```

<br>

3. Run GenAI-Perf (on Triton Server). ***Allow for approx. 30sec for the script to finish running.***
    
    ```bash
    export INPUT_SEQUENCE_LENGTH=200
    export INPUT_SEQUENCE_STD=10
    export OUTPUT_SEQUENCE_LENGTH=200
    export CONCURRENCY=10
    export MODEL=meta/llama-3.1-8b-instruct
    
    cd /workdir
    genai-perf \
        -m $MODEL \
        --endpoint-type chat \
        --service-kind openai \
        --streaming \
        -u localhost:8000 \
        --synthetic-input-tokens-mean $INPUT_SEQUENCE_LENGTH \
        --synthetic-input-tokens-stddev $INPUT_SEQUENCE_STD \
        --concurrency $CONCURRENCY \
        --output-tokens-mean $OUTPUT_SEQUENCE_LENGTH \
        --extra-inputs max_tokens:$OUTPUT_SEQUENCE_LENGTH \
        --extra-inputs min_tokens:$OUTPUT_SEQUENCE_LENGTH \
        --extra-inputs ignore_eos:true \
        --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
        -- \
        -v \
        --max-threads=256
    ```
    Sample Output: <br>
    ![image](https://github.com/ymlee4953/Dell-DemoLab/blob/main/DemoRoom/NVIDIA%20GenAI%20Workshop/images/lab1-genai-perf.png)

<br>

4. Exit out of the container.    
    
    ```bash
    exit
    ```

<br>

>**ADDITIONAL READING**
>
>Performing Benchmark Sweep for various use cases: 
>
>- https://docs.nvidia.com/nim/benchmarking/llm/latest/step-by-step.html#step-4-sweeping-through-a-number-of-use-cases
