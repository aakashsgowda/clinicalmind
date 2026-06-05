import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        database=os.getenv("POSTGRES_DB", "clinicalmind"),
        user=os.getenv("POSTGRES_USER", "clinicalmind"),
        password=os.getenv("POSTGRES_PASSWORD", "clinicalmind123")
    )

def clean_text(text):
    if pd.isna(text):
        return None
    return str(text).strip()

def ingest_mtsamples():
    print("=== Starting MTSamples Ingestion ===")
    
    df = pd.read_csv("data/raw/mtsamples.csv")
    print(f"Loaded {len(df)} records from MTSamples")
    
    df = df.dropna(subset=['transcription'])
    print(f"After dropping missing transcriptions: {len(df)} records")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("TRUNCATE TABLE raw.clinical_notes RESTART IDENTITY CASCADE")
    conn.commit()
    print("Cleared existing raw data")
    
    records = []
    for _, row in df.iterrows():
        records.append((
            clean_text(row.get('description')),
            clean_text(row.get('medical_specialty')),
            clean_text(row.get('sample_name')),
            clean_text(row.get('transcription')),
            clean_text(row.get('keywords'))
        ))
    
    insert_query = """
        INSERT INTO raw.clinical_notes 
        (description, medical_specialty, sample_name, transcription, keywords)
        VALUES (%s, %s, %s, %s, %s)
    """
    
    print(f"Inserting {len(records)} records...")
    execute_batch(cursor, insert_query, records, page_size=100)
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM raw.clinical_notes")
    count = cursor.fetchone()[0]
    print(f"✅ Successfully ingested {count} records into raw.clinical_notes")
    
    cursor.close()
    conn.close()
    return count

if __name__ == "__main__":
    ingest_mtsamples()
