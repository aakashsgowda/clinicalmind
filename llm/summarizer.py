import os
from openai import OpenAI
from dotenv import load_dotenv
import psycopg2

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        database=os.getenv("POSTGRES_DB", "clinicalmind"),
        user=os.getenv("POSTGRES_USER", "clinicalmind"),
        password=os.getenv("POSTGRES_PASSWORD", "clinicalmind123")
    )

def summarize_clinical_note(transcription, specialty=None):
    """Summarize a clinical note into structured format"""
    
    system_prompt = """You are a clinical documentation specialist. 
    Your task is to summarize clinical notes into a structured format.
    Always respond in exactly this format:

    CHIEF COMPLAINT: [main reason for visit]
    DIAGNOSIS: [primary diagnosis or procedure]
    KEY FINDINGS: [2-3 most important clinical findings]
    MEDICATIONS: [medications mentioned, or 'None mentioned']
    PROCEDURES: [procedures performed, or 'None mentioned']
    FOLLOW-UP: [follow-up instructions, or 'Not specified']
    SUMMARY: [1-2 sentence plain English summary]

    Be concise and precise. Use medical terminology appropriately."""
    
    user_prompt = f"""Please summarize this clinical note:

Specialty: {specialty or 'Unknown'}

Clinical Note:
{transcription[:3000]}"""  # Limit to 3000 chars to save tokens
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500,
        temperature=0.1  # Low temperature for consistent medical summaries
    )
    
    return response.choices[0].message.content

def summarize_notes_by_specialty(specialty, limit=3):
    """Summarize multiple notes from a specific specialty"""
    print(f"\n=== Summarizing {specialty} Notes ===")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, sample_name, transcription, medical_specialty
        FROM clean.clinical_notes
        WHERE medical_specialty = %s
        AND word_count BETWEEN 100 AND 500
        LIMIT %s
    """, (specialty, limit))
    
    notes = cursor.fetchall()
    cursor.close()
    conn.close()
    
    summaries = []
    for note in notes:
        note_id, sample_name, transcription, medical_specialty = note
        print(f"\nSummarizing: {sample_name}")
        print("-" * 50)
        
        summary = summarize_clinical_note(transcription, medical_specialty)
        print(summary)
        
        summaries.append({
            'note_id': note_id,
            'sample_name': sample_name,
            'specialty': medical_specialty,
            'summary': summary
        })
    
    return summaries

if __name__ == "__main__":
    # Test with different specialties
    specialties = ["Cardiovascular / Pulmonary", "Orthopedic", "Neurology"]
    
    all_summaries = []
    for specialty in specialties:
        summaries = summarize_notes_by_specialty(specialty, limit=1)
        all_summaries.extend(summaries)
    
    print(f"\n✅ Successfully summarized {len(all_summaries)} clinical notes!")