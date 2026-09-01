from modelscope import snapshot_download

model_dir = snapshot_download(
    'Qwen/Qwen3-8B',
    local_dir='./model/qwen'
)

model_dir = snapshot_download(
    'LLM-Research/Meta-Llama-3.1-8B-Instruct',
    local_dir='./model/llama'
)
