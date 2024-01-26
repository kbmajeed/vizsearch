import io
import os
import logging
import zipfile

import boto3
import botocore
import dotenv
import yaml

from utils import initialize


# Load config files and env variables
config = initialize.load_config()
_ = dotenv.load_dotenv()


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
            bucket.download_fileobj(config.dataset.AWS_S3_Dataset, data)
            logging.info(f'Reaching AWS S3 for: {config.dataset.AWS_S3_Dataset} ')
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                logging.warning("The object does not exist.")
            else:
                logging.warning("Something went wrong.")
                raise

        # try:
        #     bucket.download_fileobj(config.dataset.AWS_S3_Dataset, data)
        #     logging.info(f'Reaching AWS S3 for: {config.dataset.AWS_S3_Dataset} ')
        # except botocore.exceptions.ClientError as e:
        #     if e.response['Error']['Code'] == "404":
        #         print("The object does not exist.")
        #     else:
        #         print("Something went wrong.")
        #         raise
    pass


if __name__ == "__main__":
    _ = initialize.load_logging()
    fetch_raw_datafiles_from_s3()

