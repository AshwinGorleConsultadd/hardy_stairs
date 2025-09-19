"""
Simple Entry Point for Video Processing Pipeline

This file serves as a simple entry point that calls the main processing function.
"""

from video_analyser_service import process_video_and_generate_report

def main():
    """Simple entry point - just call the function and get report URL"""
    
    #Example 1: Local video without S3 upload
    # report_url = process_video_and_generate_report(
    #     video_source_type="local",
    #     video_url="input/part000.mp4",
    #     upload_to_s3=False
    # )
    # print("Report URL:", report_url)
    
    # Example 2: Local video with S3 upload
    report_url = process_video_and_generate_report(
        video_source_type="s3",
        video_url="https://hardymfg.s3.ap-south-1.amazonaws.com/part-2+trimmed.mp4",
        presigned_s3_url="https://hardymfg.s3.amazonaws.com/string?uploadId=k765GFJugmnhFXODVvGfturU1fvg3NJwVjfCnyq7iuUw575CCJ9Km5bxBhN4RLiCZvLiFCcsFIHugVAA6M0NVfjx8XUMVh3b8YmWoA1rwtL_DQLuXIOT_pPLOi3nguct&partNumber=1&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAR6FXMOHR3QHQ27KX%2F20250919%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20250919T000644Z&X-Amz-Expires=36000&X-Amz-SignedHeaders=host&X-Amz-Signature=a049f5d4e067ad55c8a6f53824822c2e2f03934ac73fb5827857ef5dc924c592",
        upload_to_s3=True
    )
    print("Report URL:", report_url)

if __name__ == "__main__":
    main()