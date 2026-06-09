import os
from openai import OpenAI
from dotenv import load_dotenv
from retrieval.hybrid_search import HybridSearch

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ClinicalQAEngine:
    def __init__(self):
        print("Initializing Clinical Q&A Engine...")
        self.searcher = HybridSearch()
        print("✅ Q&A Engine ready!")
    
    def answer_question(self, question, n_chunks=5):
        """Answer a clinical question using RAG"""
        
        # Step 1: Retrieve relevant chunks
        results = self.searcher.hybrid_search(question, n_results=n_chunks)
        
        if not results:
            return {
                'answer': 'No relevant clinical notes found.',
                'sources': [],
                'chunks_used': 0
            }
        
        # Step 2: Build context from retrieved chunks
        context = ""
        sources = []
        
        for i, result in enumerate(results):
            context += f"\n--- Clinical Note {i+1} ---\n"
            context += f"Specialty: {result['metadata']['specialty']}\n"
            context += f"Sample: {result['metadata']['sample_name']}\n"
            context += f"Content: {result['document']}\n"
            
            source = f"{result['metadata']['sample_name']} ({result['metadata']['specialty']})"
            if source not in sources:
                sources.append(source)
        
        # Step 3: Generate answer with GPT-4o-mini
        system_prompt = """You are a clinical information assistant helping 
        healthcare professionals find relevant information from medical records.
        
        Rules:
        - Answer based ONLY on the provided clinical notes
        - Be specific and cite which note supports your answer
        - If the notes don't contain relevant information, say so clearly
        - Use medical terminology appropriately
        - Keep answers concise and actionable
        - Always mention the source note"""
        
        user_prompt = f"""Based on the following clinical notes, please answer this question:

Question: {question}

Clinical Notes:
{context}

Please provide a clear, accurate answer based only on the information in these notes."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=600,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'chunks_used': len(results)
        }
    
    def print_answer(self, result):
        print(f"\n{'='*60}")
        print(f"Q: {result['question']}")
        print(f"{'='*60}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources ({result['chunks_used']} chunks used):")
        for source in result['sources']:
            print(f"  • {source}")

if __name__ == "__main__":
    qa = ClinicalQAEngine()
    
    questions = [
        "What medications are commonly prescribed after knee replacement surgery?",
        "What are the key findings in cardiac catheterization procedures?",
        "What follow-up care is recommended after orthopedic procedures?"
    ]
    
    for question in questions:
        result = qa.answer_question(question, n_chunks=5)
        qa.print_answer(result)
    
    print(f"\n✅ Q&A Engine test complete!")