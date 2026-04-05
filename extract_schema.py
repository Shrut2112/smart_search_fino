import psycopg2

def extract_schema():
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres.cxoartbafydqirizvyzh",
        password="Codeis@04fino",
        host="aws-1-ap-southeast-2.pooler.supabase.com",
        port=6543,
        sslmode="require"
    )
    
    with conn.cursor() as cur:
        # Get all tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [r[0] for r in cur.fetchall()]
        
        for table in tables:
            print(f"-- TABLE: {table}")
            cur.execute("""
                SELECT column_name, data_type, character_maximum_length, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, (table,))
            
            columns = cur.fetchall()
            for col in columns:
                print(col)
                
            print("---")

if __name__ == "__main__":
    extract_schema()
