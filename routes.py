import json
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from embeddings import get_embeddings
from faiss_index import faiss_manager
from models import AddVectorInput, IndexMetadata, QueryInput
from config import DEFAULT_MODEL


router = APIRouter()


@router.post("/add-text")
def add_text(data: AddVectorInput, db: Session = Depends(get_db)):
    vector = get_embeddings([doc.text for doc in data.documents], DEFAULT_MODEL)
    new_ids = faiss_manager.add_vectors(vector)
    extra_metadatas = [json.dumps(doc.metadata) for doc in data.documents]

    for doc, id, extra in zip(data.documents, new_ids, extra_metadatas):
        metadata = IndexMetadata(
            id=id, text=doc.text, model_name=DEFAULT_MODEL, extra_metadata=extra
        )
        db.add(metadata)
        db.commit()

    return {"success": True, "ids": new_ids}


@router.post("/search")
def search(data: QueryInput, db: Session = Depends(get_db)):
    vector = get_embeddings([data.query], DEFAULT_MODEL)
    results = faiss_manager.search(vector)

    final_results = []

    for indice, distance in results:
        if indice == -1:
            continue

        metadata = (
            db.query(IndexMetadata).filter(IndexMetadata.id == indice.item()).first()
        )

        parsed_result = {
            "id": metadata.id,
            "distance": distance.item(),
            "text": metadata.text,
            "modelName": metadata.model_name,
            "metadata": json.loads(metadata.extra_metadata),
        }

        final_results.append(parsed_result)

    return {"results": final_results}
