1. Embedding : python embedding_model_2.py

2. Build FAISS index : python -c "from app.db.vector_db import build_and_save_indexes_v2; build_and_save_indexes_v2('data/embeddings_v2','data/faiss_index_v2')"

3. Retrieval text->frame : python scripts/search_text_frame.py "Bien canh báo nguy hiem ve sat lo, phát sóng trên HTV9 HD lúc 06:31:31, có ticker chay phía duoi" --use-llm --embeddings-root data/embeddings_v2 --index-dir data/faiss_index_v2


# UPDATE RETRIEVAL INTEGRATED W FRONTEND GUILD

1. **At least you must have the media_info directory (storing links Youtube) in /data to Preview the YT video of resulted frames !**

2. - In one terminal tab : cd backend -> uvicorn app.main:app --reload

   - In another terminal tab : cd frontend -> npm install (*not necessary if did it before*) -> npm run dev -> go to localhost:3000

3. ## In Section 1 : 
   - Retrieval tab, enter English query (**must be English cuz i turned the LLM service off ^^**) -> (optionally) adjust attributes -> perform Search

   - After receiving results -> Click on image to trigger the Image Preview Modal (*IF YOU HAVE KEYFRAMES ^^*) or Click on "Preview Frame" button to trigger the Video Preview Modal to view the Youtube video of that frame (*IF YOU ACTUALLY HAVE MEDIA_INFO ^^*).



   > PEACE.