import logging
import os
import pickle
import warnings

import dotenv
from pinecone import Pinecone, PodSpec

from utils import initialize


# Load config files and env variables
initialize.load_logging()
config = initialize.load_config()
dotenv.load_dotenv()
warnings.filterwarnings("ignore")

# Load CNN embedding model
cat_dogs_embeddings = pickle.load(open('../model/embedding_matrix.emb', 'rb'))

# Connect to Pinecone cloud vector database
pc = Pinecone(api_key=os.environ['PINECONE_APPKEY'])

# Create an index
pc.create_index(
    name="vizsearch_index",
    dimension=512,
    metric="euclidean",
    spec=PodSpec(
        environment='us-west-2',
        pod_type='p1.x1'
    )
)

index = pc.Index('vizsearch_index')

# Insert index
upsert_response = index.upsert(vectors=cat_dogs_embeddings, namespace='vizsearch-namespace')


if __name__ == '__main__':
    print(0)