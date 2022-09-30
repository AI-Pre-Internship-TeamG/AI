import boto3
import uuid
from .s3Config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET_NAME, AWS_S3_DOMAIN

def s3_connection():
    try:
        s3 = boto3.resource(
        service_name='s3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!")
        return s3

def saveImageToS3(uploadFile, state): 
    fileFormat = uploadFile.content_type.split("/")[1]
    fname = (str(uuid.uuid4()))
    key = f"{state}/{fname}" # 사진이 저장될 경로 설정
    s3r = s3_connection()
    s3r.Bucket(AWS_S3_BUCKET_NAME).put_object(Key=key, Body=uploadFile.read(), ContentType=fileFormat)
    imageKey = key
    return imageKey