import os
import pickle
from pinecone import Pinecone

import logging
import warnings

import dotenv

from utils import initialize


# Load config files and env variables
initialize.load_logging()
config = initialize.load_config()
dotenv.load_dotenv()
warnings.filterwarnings("ignore")

cat_dogs_embeddings = pickle.load(open('../model/embedding_matrix.emb', 'rb'))

pc = Pinecone(api_key=os.environ['PINECONE_APPKEY'])

# Turn off Z-scaler
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

from pinecone import Pinecone, PodSpec

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

upsert_response = index.upsert(vectors=cat_dogs_embeddings, namespace='vizsearch-namespace')



if __name__ == '__main__':

    print(0)