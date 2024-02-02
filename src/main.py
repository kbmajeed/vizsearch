import os
import pickle
import psycopg2
import logging
import warnings
import matplotlib.pyplot as plt

import imageio.v3 as iio
import dotenv
from utils import initialize


# Load config files and env variables
initialize.load_logging()
config = initialize.load_config()
dotenv.load_dotenv()
warnings.filterwarnings("ignore")

cat_dogs_embeddings = pickle.load(open('../model/embedding_matrix.emb', 'rb'))

conn = psycopg2.connect(
    database=os.environ['POSTGRES_DB'],
    user=os.environ['POSTGRES_USER'],
    password=os.environ['POSTGRES_PSWD'],
    host=os.environ['POSTGRES_HOST'],
    port=os.environ['POSTGRES_PORT']
)

cursor = conn.cursor()

embedding = cat_dogs_embeddings['cat.0.jpg']

embedding = embedding.numpy().squeeze().tolist()
sql_query_vector_embedding = "SELECT * FROM vizsearch ORDER BY embedding <-> '{query_embedding}' LIMIT {limit};"
cursor.execute(sql_query_vector_embedding.format(query_embedding=embedding, limit=3))

result = cursor.fetchall()

home_dir = '../data/raw_data/dogs-vs-cats/train/'
for item in range(len(result)):
    img_dir = os.path.join(home_dir, result[item][1])
    im = iio.imread(img_dir)
    plt.imshow(im)
    plt.show()


if __name__ == '__main__':
    print(0)