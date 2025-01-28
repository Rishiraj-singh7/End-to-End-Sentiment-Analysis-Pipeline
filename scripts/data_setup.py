import os
import sqlite3
import pandas as pd

# Paths - using raw strings
DATA_PATH = os.path.abspath(r"C:\Users\Rishi\Desktop\Data-sci\sentiment analysis pipeline\data\IMDB_Dataset.csv")
DB_PATH = os.path.abspath(r"C:\Users\Rishi\Desktop\Data-sci\sentiment analysis pipeline\database\imdb_reviews.db")


os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def setup_database():
   
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    # Connect to SQLite database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS imdb_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT,
            sentiment TEXT
        );
    """)

    # Load data into DataFrame
    df = pd.read_csv(DATA_PATH)

    # Insert data into table
    df.to_sql("imdb_reviews", conn, if_exists="replace", index=False)
    print("Data inserted successfully!")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    setup_database()
