from typing import List
import requests
import os
from config import *
import numpy as np

from models import TextInput


def get_embeddings(documents: List[str], model="jina-embeddings-v3"):

    url = JINAI_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f'Bearer {os.getenv("JINAI_API_KEY")}',
    }

    data = {
        "model": model,
        "task": "text-matching",
        "late_chunking": False,
        "dimensions": 1024,
        "embedding_type": "float",
        "input": documents,
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        data = response.json()
        arrs = np.array(
            [np.array(x["embedding"], dtype=np.float32) for x in data["data"]]
        )
        return arrs
    else:
        raise Exception(f"Failed to get embeddings: {response.text}")
