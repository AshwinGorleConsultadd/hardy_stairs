"""
Video Processing Script for Stair Repair Defect Detection

This script processes videos of building inspections to extract defect information
from audio transcripts and generate structured data for repair documentation.
"""

import os
import json
import logging
import re
from typing import List, Optional, Dict, Any
from config import AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,AWS_REGION
from transcript_refiner import refine_transcript_chunks_symmentically
from defect_description_extracter import extract_defect_from_chunk, calculate_screenshot_timestamp
from models import DefectInfo
from fuzzywuzzy import fuzz
import string
import boto3
# Audio processing imports
import whisper
import ffmpeg
from urllib.parse import urlparse

# AWS imports removed - using HTTP download for S3 instea
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Main class for processing repair videos and extracting defect information"""
    
    def __init__(self, whisper_model_name: str = "medium.en"):
        """
        Initialize the video processor
        
        Args:
            whisper_model_name: Whisper model to use for transcription
        """
        self.whisper_model_name = whisper_model_name
        self.whisper_model = None
        self.s3_client = None
        
        # Building tracking
        self.building_counter = 0
        self.current_building_name = None
        self.current_apartment_number = None
        
        # S3 client not needed - we'll use HTTP download for public URLs
        self.s3_client = None
    
    def load_whisper_model(self):
        """Load the Whisper model for transcription"""
        try:
            logger.info("🎤 Loading Whisper model: %s", self.whisper_model_name)
            
            # Check if CUDA is available
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("🖥️  Using device: %s", device)
            
            # Load model with device specification
            self.whisper_model = whisper.load_model(self.whisper_model_name, device=device)
            logger.info("✅ Whisper model '%s' loaded successfully on %s", self.whisper_model_name, device)
            
            # Log GPU memory usage if available
            if device == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1024**3
                logger.info("💾 GPU Memory used: %.2f GB", memory_used)
        except Exception as e:
            logger.error("❌ Failed to load Whisper model '%s': %s", self.whisper_model_name, e)
            raise
    
    def get_video_from_local(self, video_path: str) -> str:
        """
        Load video from local path
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the video file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info("Loading video from local path: %s", video_path)
        return video_path
    
    def get_video_from_s3(self, s3_url: str, local_path: str) -> str:
        """
        Download video from S3 using public URL
        
        Args:
            s3_url: Public S3 URL (e.g., https://bucket.s3.amazonaws.com/path/video.mp4)
            local_path: Local path to save the video
            
        Returns:
            Path to the downloaded video file
        """
        try:
            s3_client = boto3.client(
                                "s3",
                                aws_access_key_id=AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                region_name=AWS_REGION
                            )


            # Handle both s3:// and https:// URLs
            if s3_url.startswith("s3://"):
                parsed = urlparse(s3_url)
                bucket = parsed.netloc
                key = parsed.path.lstrip("/")
            else:
                # Assume https://bucket.s3.region.amazonaws.com/key format
                parsed = urlparse(s3_url)
                bucket = parsed.netloc.split(".")[0]
                key = parsed.path.lstrip("/")

            logger.info("Downloading s3://%s/%s -> %s", bucket, key, local_path)

            # Download
            s3_client.download_file(bucket, key, local_path)

            logger.info("Video downloaded to: %s", local_path)
            return local_path
            
        except Exception as e:
            logger.error("Failed to download video from S3 URL: %s", e)
            raise
    
    def extract_audio_from_video(self, video_path: str, output_audio_path: str) -> str:
        """
        Extract audio from video using FFmpeg with enhanced preprocessing
        
        Args:
            video_path: Path to the input video
            output_audio_path: Path to save the extracted audio
            
        Returns:
            Path to the extracted audio file
        """
        try:
            
            logger.info("🎬 Extracting audio from video: %s", video_path)
            
            # Enhanced audio extraction with preprocessing for better Whisper performance
            (
                ffmpeg
                .input(video_path)
                .output(
                    output_audio_path, 
                    acodec='pcm_s16le',  # 16-bit PCM
                    ac=1,  # Mono channel
                    ar='16000',  # 16kHz sample rate
                    af='highpass=f=80,lowpass=f=8000,volume=1.2'  # Audio filters for better quality
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            logger.info("✅ Audio extracted and preprocessed to: %s", output_audio_path)
            return output_audio_path
            
        # except ffmpeg.Error as e:
        #     logger.error("FFmpeg error: %s", e)
        #     raise
        except Exception as e:
            logger.error("Failed to extract audio: %s", e)
            raise
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcription result with segments and timestamps
        """
        if not self.whisper_model:
            self.load_whisper_model()
        
        try:
            logger.info("🎵 Starting transcription of audio: %s", audio_path)
            logger.info("🤖 Using Whisper model: %s", self.whisper_model_name)
            
            # Optimized parameters for better accuracy
            result = self.whisper_model.transcribe(
                audio_path, 
                word_timestamps=True, 
                language="en", 
                condition_on_previous_text=True,  # Better context awareness
                no_speech_threshold=0.3,  # Lower threshold to catch more speech
                logprob_threshold=-0.8,  # Lower threshold for better detection
                compression_ratio_threshold=2.4,  # Better compression handling
                temperature=0.0,  # Deterministic output
                best_of=1,  # Use best result
                beam_size=5,  # Required for patience parameter
                patience=1.0,  # More patience for difficult audio
                length_penalty=1.0,  # Balanced length penalty
                suppress_tokens=[-1],  # Suppress silence tokens
                initial_prompt="This is a building inspection video with technical terms about stairs, treads, defects, and construction details."  # Context prompt
            )
            
            logger.info("✅ Transcription completed successfully using model: %s", self.whisper_model_name)
            return result
        except Exception as e:
            logger.error("❌ Failed to transcribe audio with model '%s': %s", self.whisper_model_name, e)
            raise

    def extranct_required_details_from_transcript_chunks(self, transcript_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create refined transcript chunks while removing extra things from transcript object and keeping only required thinds mentioned in chunks.append
        
        Args:
            transcript_result: Whisper transcription result
            
        Returns:
            List of plain dictionaries
        """
        chunks = []
        
        for segment in transcript_result.get('segments', []):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            if text:
                # Detect building/apartment early and store in chunk for downstream preservation
                detected_building, detected_apartment = self._extract_building_apartment_from_chunk(text)
                chunks.append({
                    "description": text,
                    "start_time": start_time,
                    "end_time": end_time,
                    "building_number": detected_building,
                    "apartment_number": detected_apartment
                })
        
        logger.info("Created %d simple transcript chunks", len(chunks))
        return chunks
    
    def extract_defects_using_regs(self, refined_chunks: List[Dict[str, Any]]) -> List[DefectInfo]:
        """
        It takes refined transcript chunks and extract defects using rules and also detect building and apartment numbers and assign them to each defect.
        Rule-based defect extraction with sequential building/apartment tracking
        """
        defects = []
        # Initialize tracking variables
        last_building_number = None
        last_apartment_number = None
        building_counter = 0
        
        logger.info("Processing %d chunks sequentially for building/apartment tracking", len(refined_chunks))
        
        # Safety check for empty or None list
        if not refined_chunks:
            logger.warning("⚠️ No refined chunks to process")
            return []
        
        # Process chunks sequentially - track context AND extract defects in one pass
        for i, chunk in enumerate(refined_chunks):
            # Safety check for None chunks
            if chunk is None:
                logger.warning("⚠️ 🔴Skipping None chunk at index %d", i)
                continue
            
            # Safety check for missing description
            if not isinstance(chunk, dict) or "description" not in chunk:
                logger.warning("⚠️ 🔴Skipping invalid chunk at index %d: %s", i, chunk)
                continue
            
            description = chunk["description"]
            if not description:
                logger.warning("⚠️ 🔴Skipping chunk with empty description at index %d", i)
                continue
                
            logger.info("📝 Processing chunk %d/%d: %s", i+1, len(refined_chunks), description[:100])
            logger.debug("🔍 Chunk data: %s", chunk)
            
            # Prefer numbers provided by upstream (LLM-refined or initial chunk); fallback to in-place extraction
            provided_building = chunk.get("building_number") if isinstance(chunk, dict) else None
            provided_apartment = chunk.get("apartment_number") if isinstance(chunk, dict) else None
            if provided_building or provided_apartment:
                building_num, apartment_num = provided_building, provided_apartment
            else:
                building_num, apartment_num = self._extract_building_apartment_from_chunk(description)
            
            if building_num:
                building_counter += 1
                last_building_number = building_num
                logger.info("🏢 Building detected: %s -> building%d", description.strip(), building_counter)
            
            if apartment_num:
                last_apartment_number = apartment_num
                logger.info("🏠 Apartment detected: %s", apartment_num)
            
            # Check if this chunk contains defect information
            description_lower = description.lower()
            if any(keyword in description_lower for keyword in ['tread', 'track', 'try', 'tri', 'tred', 'thread']):
                try:
                    defect_info = extract_defect_from_chunk(chunk)
                    if defect_info:
                        # Assign the current building/apartment numbers (not the final ones)
                        defect_info.building_counter = f"building{building_counter}" if building_counter > 0 else None
                        defect_info.apartment_number = last_apartment_number
                        defect_info.ss_timestamp = calculate_screenshot_timestamp(defect_info.timestamp_start, defect_info.timestamp_end)
                        defects.append(defect_info)
                        logger.info("🔍 Defect extracted: %s (Building: %s, Apartment: %s)", 
                                   defect_info.description[:50], defect_info.building_counter, defect_info.apartment_number)
                except Exception as e:
                    logger.error("❌ Error extracting defect from chunk %d: %s", i, e)
                    logger.error("   Chunk data: %s", chunk)
        
        logger.info("Extracted %d defects using rule-based method", len(defects))
        
        # Log summary of building/apartment tracking
        logger.info("📊 Final tracking summary:")
        logger.info("   🏢 Building counter: %d", building_counter)
        logger.info("   🏠 Last apartment number: %s", last_apartment_number)
        logger.info("   📋 Defects extracted: %d", len(defects))
        
        return defects
    
    def _clean_word(self, word: str) -> str:
        """Clean word by removing punctuation and converting to lowercase"""
        return word.strip(string.punctuation).lower()
    
    def _fuzzy_find(self, word: str, target: str, threshold: int = 80) -> bool:
        """Check if word matches target using fuzzy matching"""
        word = self._clean_word(word)
        return fuzz.ratio(word, target.lower()) >= threshold
    
    def _extract_building_apartment_from_chunk(self, description: str) -> tuple[Optional[str], Optional[str]]:
        """Extract building and apartment numbers from chunk description using fuzzy matching"""
        tokens = description.split()
        building_num = None
        apartment_num = None
        
        # Also check for numbers that might be standalone
        standalone_numbers = re.findall(r'\b(\d+)\b', description)
        
        for i, token in enumerate(tokens):
            # Check for building keywords
            if any(self._fuzzy_find(token, kw) for kw in ["building", "department", "andepartment", "bolding", "depart", "bldg"]):
                if i + 1 < len(tokens):
                    num = re.sub(r"\D", "", tokens[i + 1])
                    if num.isdigit():
                        building_num = num
                        logger.info("🔍 Found building number: %s after keyword: %s", num, token)
            
            # Check for apartment keywords
            if any(self._fuzzy_find(token, kw) for kw in ["apartment", "aprtmen", "apt", "apt.", "unit"]):
                if i + 1 < len(tokens):
                    num = re.sub(r"\D", "", tokens[i + 1])
                    if num.isdigit():
                        apartment_num = num
                        logger.info("🔍 Found apartment number: %s after keyword: %s", num, token)
        
        # If we found keywords but no numbers, try to find numbers in the same chunk
        if not building_num and any(self._fuzzy_find(token, kw) for token in tokens for kw in ["building", "department", "andepartment", "bolding", "depart", "bldg"]):
            for num in standalone_numbers:
                if len(num) >= 2:  # Building numbers are usually 2+ digits
                    building_num = num
                    logger.info("🔍 Found building number in chunk: %s", num)
                    break
        
        if not apartment_num and any(self._fuzzy_find(token, kw) for token in tokens for kw in ["apartment", "aprtmen", "apt", "apt.", "unit"]):
            for num in standalone_numbers:
                if len(num) >= 2:  # Apartment numbers are usually 2+ digits
                    apartment_num = num
                    logger.info("🔍 Found apartment number in chunk: %s", num)
                    break
        
        return building_num, apartment_num
    
    #-------------------------------------------------------------------------------
    def _filter_relevant_chunks(self, refined_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter chunks that contain relevant defect information"""
        relevant_keywords = [
            'building', 'apartments', 'apartment', 'department',
            'track', 'try', 'tri', 'tred', 'treads', 'thread', 'tread',
            'priority', 'one', 'two', 'three', '1', '2', '3',
            'rear', 'front', 'top', 'bottom', 'center',
            'crack', 'cracks', 'defect', 'defects', 'damage', 'to'
        ]
        
        relevant_chunks = []
        for chunk in refined_chunks:
            description_lower = chunk["description"].lower()
            
            # Check if chunk contains any relevant keywords
            if any(keyword in description_lower for keyword in relevant_keywords):
                relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    
    def _parse_defect_response(self, response: str) -> List[DefectInfo]:
        """Parse LLM response manually to extract DefectInfo objects"""
        try:
            import json
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                defects = []
                for item in data:
                    if isinstance(item, dict):
                        defect = DefectInfo(
                            building_counter=item.get('building_counter'),
                            building_name=item.get('building_name'),
                            apartment_number=item.get('apartment_number'),
                            tread_number=item.get('tread_number'),
                            priority=item.get('priority'),
                            description=item.get('description'),
                            timestamp_start=item.get('timestamp_start'),
                            timestamp_end=item.get('timestamp_end'),
                            transcript_segment=item.get('transcript_segment')
                        )
                        defects.append(defect)
                
                return defects
            
            # Fallback: try to parse as simple text
            return self._parse_simple_defect_response(response)
            
        except Exception as e:
            logger.error("Failed to parse defect response: %s", e)
            return []
    
    def _parse_simple_defect_response(self, response: str) -> List[DefectInfo]:
        """Fallback parsing for simple text responses"""
        defects = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and ('tread' in line.lower() or 'priority' in line.lower() or 'crack' in line.lower()):
                # Extract basic information using regex
                tread_match = re.search(r'tread\s+(?:number\s+)?(\d+)', line.lower())
                priority_match = re.search(r'priority\s+(one|two|three|four|five|\d+)', line.lower())
                description_match = re.search(r'(top|bottom|front|rear|center)\s+(rear|front|top|bottom|center)?\s*(crack|cracks)', line.lower())
                
                tread_number = tread_match.group(1) if tread_match else None
                priority = priority_match.group(1) if priority_match else None
                description = description_match.group(0) if description_match else None
                
                if tread_number or priority or description:
                    defect = DefectInfo(
                        tread_number=tread_number,
                        priority=priority,
                        description=description,
                        transcript_segment=line
                    )
                    defects.append(defect)
        
        return defects
    
    def process_video(self, video_source: Dict[str, str], output_dir: str = "output") -> List[DefectInfo]:
        """
        Main method to process a video and extract defect information
        
        Args:
            video_source: Dictionary with source information
                - type: "local" or "s3"
                - path: local path (for local) or S3 key (for S3)
                - bucket: S3 bucket name (for S3)
            output_dir: Directory to save intermediate files
            
        Returns:
            List of extracted defect information
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Get video
        if video_source["type"] == "local":
            video_path = self.get_video_from_local(video_source["path"])
        elif video_source["type"] == "s3":
            local_video_path = os.path.join(output_dir, "downloaded_video.mp4")
            video_path = self.get_video_from_s3(
                video_source["path"],  # Now expects S3 URL directly
                local_video_path
            )
        else:
            raise ValueError("Invalid video source type. Use 'local' or 's3'")
        
        # Step 2: Extract audio
        audio_path = os.path.join(output_dir, "extracted_audio.wav")
        self.extract_audio_from_video(video_path, audio_path)
        
        # Step 3: Transcribe audio
        transcript_result = self.transcribe_audio(audio_path)
        
        # Save transcript for debugging
        transcript_file = os.path.join(output_dir, "1_transcript.json")
        with open(transcript_file, 'w', encoding='utf-8') as f:
            json.dump(transcript_result, f, indent=2)
        
        
        # Step 4: Create refined transcript chunks
        formated_transcript_chunks = self.extranct_required_details_from_transcript_chunks(transcript_result)
        
        # Save refined transcript chunks
        refined_chunks_file = os.path.join(output_dir, "2_formated_transcript_chunks.json")
        with open(refined_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(formated_transcript_chunks, f, indent=2)

        # Step 4.6: Semantic processing of formated chunks with retry logic
        llms_refined_transcript_cunks = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info("Attempting LLM processing (attempt %d/%d)...", attempt + 1, max_retries)
                llms_refined_transcript_cunks = refine_transcript_chunks_symmentically(
                                defects_data=formated_transcript_chunks,
                                save_to_file="output/3_refined_transcript_llm.json",
                                verbose=True
                )
                
                # Check if we got valid results
                if llms_refined_transcript_cunks is not None and isinstance(llms_refined_transcript_cunks, list):
                    logger.info("✅ LLM processing successful on attempt %d", attempt + 1)
                    break
                else:
                    logger.warning("⚠️ LLM processing returned invalid result on attempt %d", attempt + 1)
                    llms_refined_transcript_cunks = None
                    
            except Exception as e:
                logger.error("❌ LLM processing failed on attempt %d: %s", attempt + 1, e)
                llms_refined_transcript_cunks = None
                
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                logger.info("⏳ Waiting %d seconds before retry...", wait_time)
                import time
                time.sleep(wait_time)
        
        # Fallback: Use original formated chunks if LLM processing fails
        if llms_refined_transcript_cunks is None:
            logger.warning("⚠️🔴🔴🔴🔴🔴 LLM processing failed after %d attempts. Using original formated chunks as fallback.", max_retries)
            llms_refined_transcript_cunks = formated_transcript_chunks
        
        # Step 5: Extract defects using refined chunks
        defects = self.extract_defects_using_regs(llms_refined_transcript_cunks)

        
        # Save defects for debugging
        defects_file = os.path.join(output_dir, "4_extracted_defects.json")
        with open(defects_file, 'w', encoding='utf-8') as f:
            json.dump([defect.model_dump() for defect in defects], f, indent=2)
        
        logger.info("Processing completed. Found %d defects.", len(defects))
        
        # Print all defects to terminal
        print("\n" + "="*80)
        print("📋 EXTRACTED DEFECTS SUMMARY")
        print("="*80)
        # for i, defect in enumerate(defects, 1):
        #     print(f"\n🔍 Defect #{i}:")
        #     print(f"  Building: {defect.building_counter} ({defect.building_name})")
        #     print(f"  Apartment: {defect.apartment_number}")
        #     print(f"  Tread: {defect.tread_number}")
        #     print(f"  Priority: {defect.priority}")
        #     print(f"  Description: {defect.description}")
        #     timestamp_start = f"{defect.timestamp_start:.2f}" if defect.timestamp_start is not None else "None"
        #     timestamp_end = f"{defect.timestamp_end:.2f}" if defect.timestamp_end is not None else "None"
        #     ss_timestamp = f"{defect.ss_timestamp:.2f}" if defect.ss_timestamp is not None else "None"
        #     print(f"  Time Range: {timestamp_start}s - {timestamp_end}s")
        #     print(f"  Screenshot Time: {ss_timestamp}s")
        #     print(f"  Transcript: {defect.transcript_segment}")
        print("="*80)
        print(f"🔵Total defects found: {len(defects)}")
        print("="*80 + "\n")
        
        # Step 6.5: Take screenshots for defects
        defects_with_image_path = []
        if defects:
            try:
                from screenshot_taker import take_screenshots_for_defects
                defects_with_image_path = take_screenshots_for_defects(defects, video_path, output_dir)
                logger.info("Screenshots captured for %d defects", len(defects_with_image_path))
            except Exception as e:
                logger.error("Failed to capture screenshots: %s", e)
                # Create defects_with_image_path without screenshots
                for defect in defects:
                    defect_dict = defect.model_dump()
                    defect_dict['image_path'] = None
                    defect_dict['image_filename'] = None
                    defects_with_image_path.append(defect_dict)

        # Step 7: Generate PDF report
        if defects_with_image_path:
            try:
                from pdf_generator import generate_pdf_report
                pdf_output_path = os.path.join(output_dir, "defects_report.pdf")
                pdf_success = generate_pdf_report(defects_with_image_path, pdf_output_path)
                if pdf_success:
                    logger.info(f"✅ PDF report generated: {pdf_output_path}")
                else:
                    logger.error("❌ Failed to generate PDF report")
            except Exception as e:
                logger.error("Failed to generate PDF report: %s", e)
        
        return defects

    def upload_pdf_to_s3(self, local_pdf_path: str, presigned_s3_url: str) -> Optional[str]:
        """
        Upload PDF report to S3 using presigned URL
        
        Args:
            local_pdf_path: Local path to the PDF file
            presigned_s3_url: Presigned S3 URL for upload
            
        Returns:
            S3 URL if successful, None if failed
        """
        try:
            import requests
            
            # Check if PDF file exists
            if not os.path.exists(local_pdf_path):
                logger.error("❌ PDF file not found: %s", local_pdf_path)
                return None
            
            # Get file size for logging
            file_size = os.path.getsize(local_pdf_path)
            file_size_mb = file_size / (1024 * 1024)
            
            logger.info("🚀 Starting S3 upload process...")
            logger.info("📁 File: %s (%.2f MB)", local_pdf_path, file_size_mb)
            
            # Read PDF file
            with open(local_pdf_path, 'rb') as f:
                pdf_data = f.read()
            
            # Upload to S3 using presigned URL
            logger.info("📤 Uploading PDF to S3...")
            logger.info("⏳ Please wait, this may take a moment...")
            
            response = requests.put(presigned_s3_url, data=pdf_data, headers={'Content-Type': 'application/pdf'})
            
            if response.status_code == 200:
                # Extract S3 URL from presigned URL (remove query parameters)
                s3_url = presigned_s3_url.split('?')[0]
                logger.info("🎉 S3 upload completed successfully!")
                logger.info("✅ PDF uploaded to: %s", s3_url)
                logger.info("📊 File size: %.2f MB", file_size_mb)
                return s3_url
            else:
                logger.error("❌ S3 upload failed with status code: %d", response.status_code)
                logger.error("📝 Response: %s", response.text)
                return None
                
        except Exception as e:
            logger.error("❌ Error uploading PDF to S3: %s", e)
            return None


def process_video_and_generate_report(
    video_source_type: str,
    video_url: str,
    presigned_s3_url: Optional[str] = None,
    upload_to_s3: bool = False,
) -> str:
    """
    Simple function to process video and return report URL
    
    Args:
        video_source_type: Type of video source ('local' or 's3')
        video_url: URL/path to the video file
        presigned_s3_url: Presigned S3 URL for uploading final report
        upload_to_s3: Whether to upload final report to S3
        openai_api_key: OpenAI API key for LLM processing
        
    Returns:
        Report URL (local or S3)
    """
    try:
        logger.info("🚀 Starting video processing...")
        
        # Initialize video processor
        processor = VideoProcessor(
            whisper_model_name="base.en"
        )
        
        # Process video and get defects
        video_source = {
            "type": video_source_type,
            "path": video_url  # For S3, this should be the public URL
        }
        defects = processor.process_video(video_source)
        
        # Get local PDF report path
        output_dir = "output"
        local_pdf_path = os.path.join(output_dir, "defects_report.pdf")
        
        # Upload to S3 if requested
        if upload_to_s3 and presigned_s3_url:
            logger.info("🌐 S3 upload requested - starting upload process...")
            s3_url = processor.upload_pdf_to_s3(local_pdf_path, presigned_s3_url)
            if s3_url:
                logger.info("🎊 S3 upload process completed successfully!")
                logger.info("🔗 Report URL: %s", s3_url)
                return s3_url
            else:
                logger.error("⚠️ S3 upload failed, returning local path instead")
                logger.info("📁 Local report URL: %s", local_pdf_path)
                return local_pdf_path
        else:
            logger.info("✅ Processing completed, returning local path")
            logger.info("📁 Local report URL: %s", local_pdf_path)
            return local_pdf_path
            
    except Exception as e:
        logger.error("❌ Video processing failed: %s", e)
        return f"Error: {str(e)}"

