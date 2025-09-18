"""
Simple Entry Point for Video Processing Pipeline

This file serves as a simple entry point that calls the main processing function.
"""

from main import process_video_and_generate_report

def main():
    """Simple entry point - just call the function and get report URL"""
    
    #Example 1: Local video without S3 upload
    report_url = process_video_and_generate_report(
        video_source_type="local",
        video_url="input/part000.mp4",
        upload_to_s3=False
    )
    print("Report URL:", report_url)
    
    # Example 2: Local video with S3 upload
    # report_url = process_video_and_generate_report(
    #     video_source_type="local",
    #     video_url="input/part000.mp4",
    #     presigned_s3_url="https://hardymfg.s3.amazonaws.com/abc.mp3?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAR6FXMOHR3QHQ27KX%2F20250918%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20250918T181634Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=0d2df31bfdf5fa8358321d093835b02147bcbfb754bc15d886db388cd72f2c90",
    #     upload_to_s3=True
    # )
    # print("Report URL:", report_url)

if __name__ == "__main__":
    main()