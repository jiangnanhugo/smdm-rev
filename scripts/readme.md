
```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="nieshen/SMDM",
    filename="mdm_safetensors/mdm-170M-100e18.safetensors",
    repo_type="model",
    local_dir="./models"
)
```