import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from embeddings.chunker import load_and_chunk_notes

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings_batch(texts, model="text-embedding-3-small"):
    """Get embeddings for a batch of texts"""
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]

def embed_chunks(chunks, batch_size=50):
    """Embed all chunks in batches"""
    print(f"\n=== Generating Embeddings ===")
    print(f"Total chunks to embed: {len(chunks)}")
    print(f"Batch size: {batch_size}")
    print(f"Model: text-embedding-3-small")
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i:i + batch_size]
        texts = [chunk['text'] for chunk in batch]
        
        try:
            embeddings = get_embeddings_batch(texts)
            all_embeddings.extend(embeddings)
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Add zero embeddings for failed batch
            all_embeddings.extend([[0] * 1536] * len(batch))
    
    print(f"\n✅ Generated {len(all_embeddings)} embeddings")
    print(f"Embedding dimensions: {len(all_embeddings[0])}")
    
    return all_embeddings

if __name__ == "__main__":
    # Load and chunk notes
    chunks = load_and_chunk_notes(limit=500)
    
    # Generate embeddings
    embeddings = embed_chunks(chunks, batch_size=50)
    
    print(f"\nSample embedding (first 5 dimensions):")
    print(embeddings[0][:5])