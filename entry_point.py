"""
Simple Entry Point for Video Processing Pipeline

This file serves as a simple entry point that calls the main processing function.
"""

from main import process_video_and_generate_report

def main():
    """Simple entry point - just call the function and get report URL"""
    
    # Example 1: Local video without S3 upload
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
    #     presigned_s3_url="https://your-bucket.s3.amazonaws.com/reports/presigned-url",
    #     upload_to_s3=True
    # )
    # print("Report URL:", report_url)

if __name__ == "__main__":
    main()