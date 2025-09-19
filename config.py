"""
Configuration file for the video processing script
"""

import os
from pathlib import Path
import boto3
# Base paths
PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
AUDIO_DIR = PROJECT_ROOT / "audio_folder"

# Create directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_DIR, AUDIO_DIR]:
    directory.mkdir(exist_ok=True)

# Whisper model configuration
WHISPER_MODEL = "base.en"  # Options: tiny, base, small, medium, large, base.en, small.en, medium.en, large.en

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# AWS S3 configuration (optional)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Video processing settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = "pcm_s16le"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Test video paths
TEST_VIDEO_LOCAL = INPUT_DIR / "videoplayback.mp4"
TEST_VIDEO_S3_BUCKET = "your-bucket-name"
TEST_VIDEO_S3_KEY = "path/to/video.mp4"

import os
import boto3
from dotenv import load_dotenv

load_dotenv()

# S3 Configuration
S3_BUCKET  = os.getenv("AWS_BUCKET_NAME")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION     = os.getenv("AWS_REGION")

# Initialize boto3 client (shared everywhere)
s3_client = boto3.client(
    "s3",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION
)
