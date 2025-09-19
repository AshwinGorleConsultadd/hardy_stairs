from config import s3_client, S3_BUCKET

def initiate_multipart_upload(file_name: str, content_type: str) -> str:
    """
    Initiates multipart upload in S3 and returns uploadId
    """
    response = s3_client.create_multipart_upload(
        Bucket=S3_BUCKET,
        Key=file_name,
        ContentType=content_type
    )
    return response["UploadId"]


def generate_presigned_url(file_name: str, upload_id: str, part_number: int) -> str:
    """
    Generates presigned URL for uploading a single part
    """
    return s3_client.generate_presigned_url(
        ClientMethod="upload_part",
        Params={
            "Bucket": S3_BUCKET,
            "Key": file_name,
            "UploadId": upload_id,
            "PartNumber": part_number
        },
        ExpiresIn=3600
    )

def generate_presigned_upload_url(file_name: str) -> str:
    """
    Generate presigned URL to upload a complete file in one request
    """
    return s3_client.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": S3_BUCKET, "Key": file_name},
        ExpiresIn=3600,
    )

def complete_multipart_upload(file_name: str, upload_id: str, parts: list[dict]) -> str:
    """
    Completes multipart upload and returns file URL
    """
    response = s3_client.complete_multipart_upload(
        Bucket=S3_BUCKET,
        Key=file_name,
        UploadId=upload_id,
        MultipartUpload={"Parts": parts},
    )
    return response["Location"]


def generate_download_url(filename: str, expires_in: int = 3600) -> str:
    """
    Generate presigned download URL
    """
    return s3_client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET, "Key": filename},
        ExpiresIn=expires_in
    )
