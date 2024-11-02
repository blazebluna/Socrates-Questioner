import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/root/autodl-tmp', revision='master')
