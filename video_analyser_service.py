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

from langchain_core.language_models import llms
from transcript_refiner import refine_transcript_chunks_symmentically
from defect_description_extracter import extract_defect_from_chunk, calculate_screenshot_timestamp
from models import DefectInfo

# Audio processing imports
import whisper
import ffmpeg

# LLM and structured output imports
try:
    from langchain_community.llms import OpenAI
    from langchain_community.prompts import PromptTemplate
    from langchain_community.output_parsers import PydanticOutputParser
except ImportError:
    try:
        from langchain.llms import OpenAI
        from langchain.prompts import PromptTemplate
        from langchain.output_parsers import PydanticOutputParser
    except ImportError:
        print("⚠️ LangChain not available. LLM features will be disabled.")
        OpenAI = None
        PromptTemplate = None
        PydanticOutputParser = None
from pydantic import BaseModel, Field

# AWS imports removed - using HTTP download for S3 instead


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Main class for processing repair videos and extracting defect information"""
    
    def __init__(self, whisper_model_name: str = "small.en", openai_api_key: Optional[str] = None):
        """
        Initialize the video processor
        
        Args:
            whisper_model_name: Whisper model to use for transcription
            openai_api_key: OpenAI API key for LLM processing
        """
        self.whisper_model_name = whisper_model_name
        self.whisper_model = None
        self.llm = None
        self.s3_client = None
        
        # Building tracking
        self.building_counter = 0
        self.current_building_name = None
        self.current_apartment_number = None
        
        # Initialize OpenAI LLM if API key provided
        if openai_api_key:
            try:
                os.environ["OPENAI_API_KEY"] = openai_api_key
                self.llm = OpenAI()
                logger.info("✅ LLM initialized successfully")
                print("✅ LLM connected successfully!")
            except Exception as e:
                logger.error("Failed to initialize LLM: %s", e)
                self.llm = None
        else:
            self.llm = None
        
        # S3 client not needed - we'll use HTTP download for public URLs
        self.s3_client = None
    
    def load_whisper_model(self):
        """Load the Whisper model for transcription"""
        try:
            logger.info("Loading Whisper model: %s", self.whisper_model_name)
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)
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
            import requests
            
            logger.info("Downloading video from S3 URL: %s", s3_url)
            
            # Download the file
            response = requests.get(s3_url, stream=True)
            response.raise_for_status()
            
            # Save to local file
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("Video downloaded to: %s", local_path)
            return local_path
            
        except Exception as e:
            logger.error("Failed to download video from S3 URL: %s", e)
            raise
    
    def extract_audio_from_video(self, video_path: str, output_audio_path: str) -> str:
        """
        Extract audio from video using FFmpeg
        
        Args:
            video_path: Path to the input video
            output_audio_path: Path to save the extracted audio
            
        Returns:
            Path to the extracted audio file
        """
        try:
            #Check if audio file already exists
            if os.path.exists(output_audio_path):
                logger.info("Audio file already exists: %s", output_audio_path)
                logger.info("Skipping audio extraction - using existing file")
                return output_audio_path
            
            logger.info("Extracting audio from video: %s", video_path)
            
            # Use ffmpeg to extract audio
            (
                ffmpeg
                .input(video_path)
                .output(output_audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            
            logger.info("Audio extracted to: %s", output_audio_path)
            return output_audio_path
            
        except ffmpeg.Error as e:
            logger.error("FFmpeg error: %s", e)
            raise
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
            logger.info("Transcribing audio: %s", audio_path)
            result = self.whisper_model.transcribe(audio_path, word_timestamps=True, language="en", condition_on_previous_text=False, no_speech_threshold=0.5 )
            logger.info("Transcription completed successfully")
            return result
        except Exception as e:
            logger.error("Failed to transcribe audio: %s", e)
            raise
    
    def _extract_building_info(self, text: str) -> Optional[Dict[str, str]]:
        """Extract building information from text"""
        
        # Patterns for building numbers (including department)
        number_patterns = [
            r'building\s+(\d+)',
            r'building\s+number\s+(\d+)',
            r'department\s+(\d+)',
            r'department\s+(\d+)\s+building\s+(\d+)'  # "department 137 building one"
        ]
        
        # Patterns for building names
        name_patterns = [
            r'building\s+([a-zA-Z][a-zA-Z0-9\s]*)',
            r'building\s+name\s+([a-zA-Z][a-zA-Z0-9\s]*)'
        ]
        
        building_info = {}
        
        # Check for building numbers (including department)
        for pattern in number_patterns:
            match = re.search(pattern, text)
            if match:
                if 'department' in pattern and 'building' in pattern:
                    # Handle "department 137 building one" case
                    building_info['number'] = match.group(2)  # building number
                else:
                    building_info['number'] = match.group(1)
                break
        
        # Check for building names (only if no number found)
        if 'number' not in building_info:
            for pattern in name_patterns:
                match = re.search(pattern, text)
                if match:
                    name = match.group(1).strip()
                    # Only consider it a name if it's not just a number
                    if not name.isdigit():
                        building_info['name'] = name
                        break
        
        return building_info if building_info else None
    
    def _extract_apartment_number(self, text: str) -> Optional[str]:
        """Extract apartment number from text"""
        patterns = [
            r'apartment\s+(\d+)',
            r'apartment\s+number\s+(\d+)',
            r'department\s+(\d+)'  # Handle misspellings
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _enrich_defects_with_timestamps(self, defects: List[DefectInfo], transcript_result: Dict[str, Any]):
        """Enrich defects with more accurate timestamp information"""
        # This method can be enhanced to provide more precise timestamps
        # based on word-level timestamps from Whisper
        # TODO: Implement word-level timestamp matching

    def create_formated_transcript_chunks(self, transcript_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create refined transcript chunks using simple chunking
        
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
                chunks.append({
                    "description": text,
                    "start_time": start_time,
                    "end_time": end_time,
                    "building_number": None,
                    "apartment_number": None
                })
        
        logger.info("Created %d simple transcript chunks", len(chunks))
        return chunks
    

    #-------------------------------------------------------------------------------
    def extract_defects_using_regs(self, refined_chunks: List[Dict[str, Any]]) -> List[DefectInfo]:
        """Rule-based defect extraction from refined chunks (fallback when LLM not available)"""
        defects = []
        
        # Extract context first
        self._extract_context_from_chunks(refined_chunks)
        
        # Filter relevant chunks
        relevant_chunks = self._filter_relevant_chunks(refined_chunks)
        logger.info("Using rule-based extraction on %d relevant chunks", len(relevant_chunks))
        
        for chunk in relevant_chunks:
            description = chunk["description"].lower()
            
            # Look for defect patterns
            if any(keyword in description for keyword in ['tread', 'track', 'try', 'tri', 'tred', 'thread']):
                defect_info = extract_defect_from_chunk(chunk)
                if defect_info:
                    defect_info.building_counter = f"building{self.building_counter}" if self.building_counter > 0 else None
                    defect_info.building_name = self.current_building_name
                    defect_info.apartment_number = self.current_apartment_number
                    defect_info.ss_timestamp = calculate_screenshot_timestamp(defect_info.timestamp_start, defect_info.timestamp_end)
                    defects.append(defect_info)
        
        logger.info("Extracted %d defects using rule-based method", len(defects))
        return defects
    
    
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
    
    def _extract_context_from_chunks(self, refined_chunks: List[Dict[str, Any]]):
        """Extract building/apartment context from all chunks"""
        # Reset context
        self.building_counter = 0
        self.current_building_name = None
        self.current_apartment_number = None
        
        for chunk in refined_chunks:
            description = chunk["description"].lower()
            
            # Extract building information
            if any(keyword in description for keyword in ['building', 'department']):
                building_info = self._extract_building_info(description)
                if building_info:
                    self.building_counter += 1
                    self.current_building_name = building_info.get('name')
                    logger.info("Building detected: %s -> building%d", chunk["description"].strip(), self.building_counter)
            
            # Extract apartment number
            if 'apartment' in description:
                apartment_match = self._extract_apartment_number(description)
                if apartment_match:
                    self.current_apartment_number = apartment_match
                    logger.info("Apartment detected: %s", apartment_match)
    
   
        """Process a batch of chunks to extract defects"""
        try:
            # Format batch chunks for LLM
            chunks_text = []
            for chunk in batch_chunks:
                chunks_text.append(f"[{chunk.start_time:.2f}s - {chunk.end_time:.2f}s] {chunk.description}")
            
            batch_text = "\n".join(chunks_text)
            
            # Create LLM prompt for defect extraction
            prompt = self._create_defect_extraction_prompt()
            
            formatted_prompt = prompt.format(transcript_chunks=batch_text)
            response = self.llm(formatted_prompt)
            
            # Parse the response manually
            defects = self._parse_defect_response(response)
            
            # Add context and calculate screenshot timestamps
            for defect in defects:
                defect.building_counter = f"building{self.building_counter}" if self.building_counter > 0 else None
                defect.building_name = self.current_building_name
                defect.apartment_number = self.current_apartment_number
                defect.ss_timestamp = self._calculate_screenshot_timestamp(defect.timestamp_start, defect.timestamp_end)
            
            return defects
            
        except Exception as e:
            logger.error("Failed to process defect batch: %s", e)
            return []
    
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
    
    
    def _create_defect_extraction_prompt(self) -> PromptTemplate:
        """Create the prompt template for defect extraction"""
        template = """
        You are an expert at extracting structured defect information from building inspection transcript chunks.
        
        Given the following transcript chunks, extract defect information with exact timestamps.
        
        EXTRACTION RULES:
        1. Only extract chunks that contain defect information (tread numbers, priorities, cracks, etc.)
        2. Use EXACT timestamps as provided in the input
        3. Extract tread numbers (may be written as "tread 9", "tread number 9", "track 9", "try 9", etc.)
        4. Extract priorities (may be "priority 1", "priority one", "priority 2", "priority two", etc.)
        5. DEFECT DESCRIPTION EXTRACTION - THIS IS THE MOST IMPORTANT PART:
           - Look for defect descriptions like "top rear crack", "bottom front crack", "top center crack", etc.
           - The description should be clean and concise (e.g., "top rear crack" not "priority one top rear crack")
           - Extract ONLY the defect description, not the entire sentence
           - Common patterns: "top rear crack", "bottom front crack", "top center crack", "bottom rear crack", "front crack", "rear crack"
           - If you see "top front rear crack" or similar, extract as "top front rear crack"
           - CRITICAL: If the entire chunk is about a defect (like "tread number eight priority one top rear crack"), 
             extract the defect description part ("top rear crack") and ignore the tread/priority information
           - NEVER leave description as null if there's a crack/defect mentioned in the chunk
        6. Handle misspellings intelligently (thread -> tread, tred -> tread, etc.)
        7. ALWAYS extract the defect description - it should contain the location (top/bottom/front/rear) and type (crack/defect)
        
        TIMESTAMP REQUIREMENTS:
        - Use EXACT start_time and end_time from input chunks
        - Do not modify or estimate timestamps
        - timestamp_start and timestamp_end should match the input exactly
        
        PRIORITY EXTRACTION:
        - Convert word priorities to numbers: "one" -> "1", "two" -> "2", "three" -> "3"
        - Keep numeric priorities as strings: "1", "2", "3"
        
        EXAMPLES:
        
        Input:
        [552.96s - 561.88s] Track number 9 priority 2 top rear crack.
        
        Output:
        {{
            "tread_number": "9",
            "priority": "2", 
            "description": "top rear crack",
            "timestamp_start": 552.96,
            "timestamp_end": 561.88,
            "transcript_segment": "Track number 9 priority 2 top rear crack."
        }}
        
        Input:
        [24.54s - 37.30s] Track, tread number 10, priority one, top rear crack, screenshot.
        
        Output:
        {{
            "tread_number": "10",
            "priority": "1",
            "description": "top rear crack",
            "timestamp_start": 24.54,
            "timestamp_end": 37.30,
            "transcript_segment": "Track, tread number 10, priority one, top rear crack, screenshot."
        }}
        
        IMPORTANT: If you see a defect mentioned but cannot extract a clean description, 
        still include the defect but try to extract the best possible description from the context.
        
        Input:
        [265.12s - 277.52s] tread number eight priority one top rear crack
        
        Output:
        {{
            "tread_number": "8",
            "priority": "1",
            "description": "top rear crack",
            "timestamp_start": 265.12,
            "timestamp_end": 277.52,
            "transcript_segment": "tread number eight priority one top rear crack"
        }}
        
        Input:
        [1021.78s - 1025.46s] fourteen priority to top center cracks
        
        Output:
        {{
            "tread_number": "14",
            "priority": null,
            "description": "top center cracks", 
            "timestamp_start": 1021.78,
            "timestamp_end": 1025.46,
            "transcript_segment": "fourteen priority to top center cracks"
        }}
        
        Transcript chunks:
        {transcript_chunks}
        
        Please provide the extracted defects in JSON format as an array of objects with the required fields.
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["transcript_chunks"]
        )
    
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
        #self.extract_audio_from_video(video_path, audio_path)
        
        # Step 3: Transcribe audio
        transcript_result = self.transcribe_audio(audio_path)
        
        # Save transcript for debugging
        transcript_file = os.path.join(output_dir, "1_transcript.json")
        with open(transcript_file, 'w', encoding='utf-8') as f:
            json.dump(transcript_result, f, indent=2)
        
        
        # Step 4: Create refined transcript chunks
        formated_transcript_chunks = self.create_formated_transcript_chunks(transcript_result)
        
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
            logger.warning("⚠️ LLM processing failed after %d attempts. Using original formated chunks as fallback.", max_retries)
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
    openai_api_key: Optional[str] = None
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
            whisper_model_name="small.en",
            openai_api_key=openai_api_key
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


def main():
    """Main function to demonstrate usage"""
    # Initialize processor
    api_key = os.getenv("OPENAI_API_KEY")
    print("API Key: ", api_key)
    
    if not api_key:
        print("⚠️  WARNING: OPENAI_API_KEY not set!")
        print("   The system will use simple chunking instead of LLM processing.")
        print("   For better results, set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print()
    
    processor = VideoProcessor(
        whisper_model_name="base.en",
        openai_api_key=api_key
    )
    
    # Example usage with local video
    video_source = {
        "type": "local",
        "path": "/Users/consultadd/Desktop/My project/consultadd_project/input/part000.mp4"
    }
    
    try:
        defects = processor.process_video(video_source)
        
        # Print results
        print(f"\nExtracted {len(defects)} defects:")
        for i, defect in enumerate(defects, 1):
            print(f"\nDefect {i}:")
            print(f"  Building Counter: {defect.building_counter}")
            print(f"  Building Name: {defect.building_name}")
            print(f"  Apartment: {defect.apartment_number}")
            print(f"  Tread: {defect.tread_number}")
            print(f"  Priority: {defect.priority}")
            print(f"  Description: {defect.description}")
            timestamp_start = f"{defect.timestamp_start:.2f}" if defect.timestamp_start is not None else "None"
            timestamp_end = f"{defect.timestamp_end:.2f}" if defect.timestamp_end is not None else "None"
            ss_timestamp = f"{defect.ss_timestamp:.2f}" if defect.ss_timestamp is not None else "None"
            
            print(f"  Timestamp: {timestamp_start}s - {timestamp_end}s")
            print(f"  Screenshot Time: {ss_timestamp}s")
            print(f"  Transcript: {defect.transcript_segment}")
            
    except Exception as e:
        logger.error("Processing failed: %s", e)
        raise


if __name__ == "__main__":
    main()
