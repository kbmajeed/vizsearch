import os
import pickle
import psycopg2
import logging
import warnings

import dotenv
from utils import initialize


# Load config files and env variables
initialize.load_logging()
config = initialize.load_config()
dotenv.load_dotenv()
warnings.filterwarnings("ignore")

# Load CNN embedding model
cat_dogs_embeddings = pickle.load(open('../model/embedding_matrix.emb', 'rb'))

# Connect to Postgres local Pgvector Vector database
conn = psycopg2.connect(
    database=os.environ['POSTGRES_DB'],
    user=os.environ['POSTGRES_USER'],
    password=os.environ['POSTGRES_PSWD'],
    host=os.environ['POSTGRES_HOST'],
    port=os.environ['POSTGRES_PORT']
)

cursor = conn.cursor()

# Execute one-time DDL command
sql_clear_vectordb_table = "DROP TABLE vizsearch;"
cursor.execute(sql_clear_vectordb_table)

sql_create_vectordb_table = "CREATE TABLE vizsearch (id bigserial PRIMARY KEY, filename VARCHAR(255), embedding vector(512));"
cursor.execute(sql_create_vectordb_table)

sql_insert_vector_embedding = "INSERT INTO vizsearch (filename, embedding) VALUES ('{filename}', '{embedding}');"

# Ingest CNN Embeddings
for ix, (filename, embedding) in enumerate(cat_dogs_embeddings.items()):
    embedding = embedding.numpy().squeeze().tolist()
    sql_ = sql_insert_vector_embedding.format(filename=filename, embedding=embedding)
    logging.info(f"Inserted embeddings for {filename} with length {len(embedding)} into pgvector:{os.environ['POSTGRES_DB']}")
    cursor.execute(sql_)

conn.commit()

if __name__ == '__main__':
    print(0)