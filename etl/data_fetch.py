import io
import os
import logging
import zipfile
import warnings

import boto3
import botocore
import dotenv

from utils import initialize


# Load config files and env variables
initialize.load_logging()
config = initialize.load_config()
dotenv.load_dotenv()
warnings.filterwarnings("ignore")


def fetch_raw_datafiles_from_s3():
    """
    Fetch raw machine learning dataset from AWS S3 bucket
    :return: None
    """

    s3 = boto3.resource(
        service_name = 's3',
        region_name = 'us-east-2',
        aws_access_key_id = os.environ['AWS_S3_ACCESS_KEY'],
        aws_secret_access_key = os.environ['AWS_S3_SECRET_ACCESS_KEY'],
    )

    bucket = s3.Bucket(config.dataset.AWS_S3_Bucket)
    with open(config.dataset.dataset_savepath, 'wb') as data:
        try:
            logging.info(f'Reaching AWS S3 for: {config.dataset.AWS_S3_Dataset} ')
            bucket.download_fileobj(config.dataset.AWS_S3_Dataset, data)
            logging.info(f'Dataset file retrieved to: {config.dataset.dataset_savepath} ')
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                logging.error("The object does not exist.")
            else:
                logging.warning("Something went wrong.")
                raise

    pass


if __name__ == "__main__":
    fetch_raw_datafiles_from_s3()

