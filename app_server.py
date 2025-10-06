from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from dotenv import load_dotenv
import os
import uuid
import sys
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from config import s3_client, S3_BUCKET
from video_analyser_service import process_video_and_generate_report
from progress import progress_store

from s3_service import (
    initiate_multipart_upload,
    generate_presigned_url,
    complete_multipart_upload,
    generate_download_url,
    generate_presigned_upload_url,
)
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --- CORS configuration for frontend access ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to frontend domain in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ------------------------
# ---- Request Models ----
# ------------------------

class InitiateUploadRequest(BaseModel):
    """
    Request body for initiating a multipart S3 upload.
    """
    fileName: str      # Name of the file to be uploaded
    contentType: str   # MIME type of the file (e.g., "video/mp4")
    partCount: int     # Number of chunks/parts for multipart upload

class CompleteUploadRequest(BaseModel):
    """
    Request body for completing a multipart S3 upload.
    """
    fileName: str      # Name of the uploaded file
    uploadId: str      # Multipart upload ID returned by initiate upload
    parts: List[Dict]  # List of parts with 'PartNumber' and 'ETag' for completion

class ProcessVideoRequest(BaseModel):
    """
    Request body for triggering video processing.
    """
    fileUrl: str       # S3 URL of the uploaded video


# ------------------------
# ---- Upload APIs ----
# ------------------------

@app.post("/initiate-upload-with-urls/")
def initiate_upload_with_urls(request: InitiateUploadRequest):
    """
    Initiate a multipart S3 upload and return presigned URLs for each part.

    **Request Body Example:**
    {
        "fileName": "inspection_video.mp4",
        "contentType": "video/mp4",
        "partCount": 5
    }

    **Response Example:**
    {
        "uploadId": "abcd1234",
        "fileName": "inspection_video.mp4",
        "presignedUrls": [
            {"partNumber": 1, "url": "..."},
            {"partNumber": 2, "url": "..."},
            ...
        ]
    }
    """
    upload_id = initiate_multipart_upload(request.fileName, request.contentType)

    def generate_url(part_number):
        return {
            "partNumber": part_number,
            "url": generate_presigned_url(request.fileName, upload_id, part_number)
        }

    # Generate presigned URLs concurrently for all parts
    with ThreadPoolExecutor() as executor:
        presigned_urls = list(executor.map(generate_url, range(1, request.partCount + 1)))

    return {
        "uploadId": upload_id,
        "fileName": request.fileName,
        "presignedUrls": presigned_urls
    }


@app.post("/complete-upload/")
def complete_upload(request: CompleteUploadRequest):
    """
    Complete a multipart S3 upload after all parts have been uploaded.

    **Request Body Example:**
    {
        "fileName": "inspection_video.mp4",
        "uploadId": "abcd1234",
        "parts": [
            {"PartNumber": 1, "ETag": "etag1"},
            {"PartNumber": 2, "ETag": "etag2"},
            ...
        ]
    }

    **Response Example:**
    {
        "fileUrl": "s3://bucket-name/inspection_video.mp4"
    }
    """
    file_url = complete_multipart_upload(
        request.fileName,
        request.uploadId,
        request.parts
    )
    return {"fileUrl": file_url}


@app.post("/process-video/")
def process_video(request: ProcessVideoRequest, background_tasks: BackgroundTasks):
    """
    Trigger processing of a video stored in S3 and generate a report CSV.

    **Request Body Example:**
    {
        "fileUrl": "s3://bucket-name/inspection_video.mp4"
    }

    **Response Example:**
    {
        "download_url": "https://s3.amazonaws.com/bucket-name/reports/1234.csv?X-Amz-Signature=..."
    }

    Steps:
    1. Generate a unique CSV report filename.
    2. Generate a presigned S3 URL for uploading the report.
    3. Run `process_video_and_generate_report` as a background task.
    4. Generate a presigned download URL for the frontend to fetch the report.
    """
   
    # Generate unique report filename
    # Generate IDs/paths first
    task_id = str(uuid.uuid4())
    report_file = f"reports/{task_id}.pdf"

    # Prepare URLs
    presigned_upload_url = generate_presigned_upload_url(report_file)
    presigned_download_url = s3_client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET, "Key": report_file},
        ExpiresIn=3600
    )

    # Initialize progress (visible immediately to clients)
    progress_store.init(task_id, message="queued")

    # Background work wrapper: marks done/error automatically
    def _runner():
        try:
            progress_store.update(task_id, status="running", message="starting…", percent=1.0)
            process_video_and_generate_report(
                video_source_type="s3",
                video_url=request.fileUrl,
                presigned_s3_url=presigned_upload_url,
                upload_to_s3=True,
                task_id=task_id,                        # <-- pass through
                progress_callback=progress_store.update # <-- pass through
            )
            # If the worker didn’t already set 100%, finalize here:
            progress_store.update(
                task_id,
                status="done",
                percent=100.0,
                message="report ready",
                result={"download_url": presigned_download_url}
            )
        except Exception as exc:
            progress_store.update(
                task_id,
                status="error",
                message=f"failed: {exc.__class__.__name__}",
            )
            # Consider logging or Sentry here

    background_tasks.add_task(_runner)

    # Return task_id so the frontend can poll /progress/{task_id}
    return {
        "task_id": task_id,
        "download_url": presigned_download_url  # optional early hint; will be valid once done
    }

@app.get("/progress/{task_id}")
def get_progress(task_id: str):
    """
    Poll the current progress of a background task.
    Response example:
    {
      "task_id": "...",
      "status": "running",
      "percent": 42.0,
      "message": "Extracting audio...",
      "started_at": 1695123456.12,
      "updated_at": 1695123499.90,
      "result": {"download_url": "..."}   # present when done
    }
    """
    data = progress_store.get(task_id)
    if not data:
        raise HTTPException(status_code=404, detail="task_id not found")
    return data

