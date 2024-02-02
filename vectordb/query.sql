sql_clear_vectordb_table = "DROP TABLE vizsearch;"

sql_create_vectordb_table = "CREATE TABLE vizsearch (id bigserial PRIMARY KEY, filename VARCHAR(255), embedding vector(512));"

sql_insert_vector_embedding = "INSERT INTO vizsearch (filename, embedding) VALUES ('{filename}', '{embedding}');"