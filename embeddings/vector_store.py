import os
import chromadb
from dotenv import load_dotenv
from embeddings.chunker import load_and_chunk_notes
from embeddings.embedder import embed_chunks

load_dotenv()

def get_chroma_client():
    return chromadb.PersistentClient(
        path="data/vectorstore"
    )

def build_vector_store(limit=500):
    print("=== Building ChromaDB Vector Store ===")
    
    # Load and chunk notes
    chunks = load_and_chunk_notes(limit=limit)
    
    # Generate embeddings
    embeddings = embed_chunks(chunks, batch_size=50)
    
    # Initialize ChromaDB
    client = get_chroma_client()
    
    # Delete existing collection if it exists
    try:
        client.delete_collection("clinical_notes")
        print("Deleted existing collection")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name="clinical_notes",
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"\nAdding {len(chunks)} chunks to ChromaDB...")
    
    # Add in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        
        collection.add(
            ids=[c['id'] for c in batch_chunks],
            embeddings=batch_embeddings,
            documents=[c['text'] for c in batch_chunks],
            metadatas=[c['metadata'] for c in batch_chunks]
        )
    
    count = collection.count()
    print(f"✅ Vector store built with {count} chunks!")
    
    return collection

def test_semantic_search(query, n_results=3):
    """Test semantic search on the vector store"""
    from embeddings.embedder import get_embeddings_batch
    
    client = get_chroma_client()
    collection = client.get_collection("clinical_notes")
    
    # Embed the query
    query_embedding = get_embeddings_batch([query])[0]
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    print(f"\n🔍 Query: '{query}'")
    print(f"Top {n_results} results:")
    print("-" * 60)
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\nResult {i+1}:")
        print(f"  Specialty: {metadata['specialty']}")
        print(f"  Sample: {metadata['sample_name']}")
        print(f"  Similarity: {1 - distance:.3f}")
        print(f"  Text preview: {doc[:200]}...")

if __name__ == "__main__":
    # Build vector store
    collection = build_vector_store(limit=500)
    
    # Test semantic search
    test_semantic_search("chest pain and heart attack treatment")
    test_semantic_search("knee replacement surgery recovery")
    test_semantic_search("diabetes medication management")