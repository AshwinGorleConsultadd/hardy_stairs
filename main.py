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


class DefectInfo(BaseModel):
    """Pydantic model for structured defect information"""
    building_counter: Optional[str] = Field(None, description="Building counter (building1, building2, etc.)")
    building_name: Optional[str] = Field(None, description="Building name if mentioned in video")
    apartment_number: Optional[str] = Field(None, description="Apartment number")
    tread_number: Optional[str] = Field(None, description="Tread number mentioned")
    priority: Optional[str] = Field(None, description="Priority level (1, 2, etc.) or (one, two, etc.)")
    description: Optional[str] = Field(None, description="Description of the defect, like (bottom rear crack, front rear crack, top front crack, top rear crack, etc.)")
    timestamp_start: Optional[float] = Field(None, description="Start timestamp in seconds")
    timestamp_end: Optional[float] = Field(None, description="End timestamp in seconds")
    ss_timestamp: Optional[float] = Field(None, description="Estimated timestamp for taking screenshot")
    transcript_segment: Optional[str] = Field(None, description="Original transcript segment")


class VideoProcessor:
    """Main class for processing repair videos and extracting defect information"""
    
    def __init__(self, whisper_model_name: str = "base.en", openai_api_key: Optional[str] = None):
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
            # Check if audio file already exists
            # if os.path.exists(output_audio_path):
            #     logger.info("Audio file already exists: %s", output_audio_path)
            #     logger.info("Skipping audio extraction - using existing file")
            #     return output_audio_path
            
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
    
    def create_llm_prompt(self) -> PromptTemplate:
        """Create the prompt template for LLM-based data extraction"""
        template = """
        You are an expert at extracting structured defect information from building inspection transcripts.
        
        Given the following transcript chunk, extract defect information. Focus on understanding the context and meaning.
        
        IMPORTANT CONTEXT UNDERSTANDING:
        - Transcript segments may be fragmented across multiple lines
        - A single defect might be split across multiple segments
        - Example: "Tread number 6 priority." + "1 top front crack." = Complete defect: "Tread number 6 priority 1 top front crack"
        - Look for patterns like: "Tread number X, priority Y"
        
        EXTRACTION RULES:
        1. Extract complete defect information by combining fragmented segments
        2. Look for tread numbers (may be written as "tread 9", "tread number 9", "tread nine", "tri 8")
        3. Look for priorities (may be written as "priority 1", "priority one", "priority 2", "priority two")
        4. Look for defect descriptions (top/bottom/front/rear + crack/defect etc)
        5. Use the timestamp from the segment that contains the main defect description
        6. Handle misspellings intelligently (thread -> tread, tred -> tread -> tri, etc.)
        
        BUILDING/APARTMENT CONTEXT:
        - If you see "apartment 111", "department 123", "building 2" - note these for context
        - These will be added automatically by the system
        Transcript chunk:
        {transcript_segments}
        
        {format_instructions}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["transcript_segments"],
            partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=List[DefectInfo]).get_format_instructions()}
        )
    
    def extract_defects_with_llm(self, transcript_result: Dict[str, Any]) -> List[DefectInfo]:
        """
        Extract defect information using LLM with structured output
        
        Args:
            transcript_result: Whisper transcription result
            
        Returns:
            List of DefectInfo objects
        """
        if  self.llm:
            logger.warning("LLM not initialized, falling back to rule-based extraction")
            return self.extract_defects_rule_based(transcript_result)
        
        try:
            # Prepare transcript segments with timestamps
            segments_text = []
            for segment in transcript_result.get('segments', []):
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '').strip()
                
                if text:
                    segments_text.append(f"[{start_time:.2f}s - {end_time:.2f}s] {text}")
            
            transcript_segments = "\n".join(segments_text)
            
            # Create prompt and parse output
            parser = PydanticOutputParser(pydantic_object=List[DefectInfo])
            prompt = self.create_llm_prompt()
            
            formatted_prompt = prompt.format(transcript_segments=transcript_segments)
            response = self.llm(formatted_prompt)
            
            # Parse the response
            defects = parser.parse(response)
            
            # Add timestamp information from original segments
            self._enrich_defects_with_timestamps(defects, transcript_result)
            
            logger.info("Extracted %d defects using LLM", len(defects))
            return defects
            
        except Exception as e:
            logger.error("LLM extraction failed: %s", e)
            logger.info("Falling back to rule-based extraction")
            return self.extract_defects_rule_based(transcript_result)
    
    
    def extract_defects_rule_based(self, transcript_result: Dict[str, Any]) -> List[DefectInfo]:
        """
        Fallback rule-based extraction when LLM is not available
        
        Args:
            transcript_result: Whisper transcription result
            
        Returns:
            List of DefectInfo objects
        """
        defects = []
        
        # Reset building tracking for new video
        self.building_counter = 0
        self.current_building_name = None
        self.current_apartment_number = None
        
        for segment in transcript_result.get('segments', []):
            text = segment.get('text', '').strip().lower()
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            # Extract building information
            if any(keyword in text for keyword in ['building', 'department']):
                building_info = self._extract_building_info(text)
                if building_info:
                    self.building_counter += 1
                    self.current_building_name = building_info.get('name')
                    logger.info("Building detected: %s -> building%d", text.strip(), self.building_counter)
            
            # Extract apartment number
            if 'apartment' in text:
                apartment_match = self._extract_apartment_number(text)
                if apartment_match:
                    self.current_apartment_number = apartment_match
            
            # Extract defect information
            if any(keyword in text for keyword in ['tread', 'thread', 'tred', 'crack', 'defect']):
                defect_info = self._extract_defect_from_text(text, start_time, end_time)
                if defect_info:
                    defect_info.building_counter = f"building{self.building_counter}" if self.building_counter > 0 else None
                    defect_info.building_name = self.current_building_name
                    defect_info.apartment_number = self.current_apartment_number
                    defect_info.transcript_segment = segment.get('text', '')
                    defects.append(defect_info)
        
        logger.info("Extracted %d defects using rule-based method", len(defects))
        return defects
    
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
    
    def _extract_defect_from_text(self, text: str, start_time: float, end_time: float) -> Optional[DefectInfo]:
        """Extract defect information from text segment"""
        
        # Look for tread patterns
        tread_patterns = [
            r'(?:tread|thread|tred)\s+(\d+)',
            r'tread\s+number\s+(\d+)'
        ]
        
        tread_number = None
        for pattern in tread_patterns:
            match = re.search(pattern, text)
            if match:
                tread_number = match.group(1)
                break
        
        # Look for priority patterns
        priority_patterns = [
            r'priority\s+(\d+)',
            r'priority\s+(one|two|three|four|five)'
        ]
        
        priority = None
        for pattern in priority_patterns:
            match = re.search(pattern, text)
            if match:
                priority = match.group(1)
                break
        
        # Look for defect descriptions
        defect_keywords = ['crack', 'defect', 'damage', 'wear', 'broken']
        description = None
        for keyword in defect_keywords:
            if keyword in text:
                # Extract surrounding context
                words = text.split()
                keyword_index = words.index(keyword) if keyword in words else -1
                if keyword_index >= 0:
                    start_idx = max(0, keyword_index - 2)
                    end_idx = min(len(words), keyword_index + 3)
                    description = ' '.join(words[start_idx:end_idx])
                    break
        
        if tread_number or priority or description:
            return DefectInfo(
                tread_number=tread_number,
                priority=priority,
                description=description,
                timestamp_start=start_time,
                timestamp_end=end_time
            )
        
        return None
    
    def _enrich_defects_with_timestamps(self, defects: List[DefectInfo], transcript_result: Dict[str, Any]):
        """Enrich defects with more accurate timestamp information"""
        # This method can be enhanced to provide more precise timestamps
        # based on word-level timestamps from Whisper
        # TODO: Implement word-level timestamp matching
    
    def _save_transcript_as_text(self, transcript_result: Dict[str, Any], output_file: str):
        """
        Save transcript as a readable text file for evaluation
        
        Args:
            transcript_result: Whisper transcription result
            output_file: Path to save the text file
        """
        try:
            logger.info("Saving transcript as text file: %s", output_file)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                # Write header
                f.write("VIDEO TRANSCRIPT\n")
                f.write("=" * 50 + "\n\n")
                
                # Write full text if available
                if 'text' in transcript_result:
                    f.write("FULL TRANSCRIPT:\n")
                    f.write("-" * 20 + "\n")
                    f.write(transcript_result['text'] + "\n\n")
                
                # Write segmented transcript with timestamps
                if 'segments' in transcript_result:
                    f.write("SEGMENTED TRANSCRIPT WITH TIMESTAMPS:\n")
                    f.write("-" * 40 + "\n")
                    
                    for i, segment in enumerate(transcript_result['segments'], 1):
                        start_time = segment.get('start', 0)
                        end_time = segment.get('end', 0)
                        text = segment.get('text', '').strip()
                        
                        # Format timestamp as MM:SS.mmm
                        start_formatted = f"{int(start_time//60):02d}:{start_time%60:06.3f}"
                        end_formatted = f"{int(end_time//60):02d}:{end_time%60:06.3f}"
                        
                        f.write(f"[{start_formatted} --> {end_formatted}] {text}\n")
                
                # Write summary
                f.write("\n" + "=" * 50 + "\n")
                f.write("TRANSCRIPT SUMMARY:\n")
                f.write(f"Total duration: {transcript_result.get('duration', 0):.2f} seconds\n")
                f.write(f"Number of segments: {len(transcript_result.get('segments', []))}\n")
                f.write(f"Language detected: {transcript_result.get('language', 'unknown')}\n")
            
            logger.info("Transcript text file saved successfully")
            
        except Exception as e:
            logger.error("Failed to save transcript text file: %s", e)
            raise
    



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
                defect_info = self._extract_defect_from_chunk(chunk)
                if defect_info:
                    defect_info.building_counter = f"building{self.building_counter}" if self.building_counter > 0 else None
                    defect_info.building_name = self.current_building_name
                    defect_info.apartment_number = self.current_apartment_number
                    defect_info.ss_timestamp = self._calculate_screenshot_timestamp(defect_info.timestamp_start, defect_info.timestamp_end)
                    defects.append(defect_info)
        
        logger.info("Extracted %d defects using rule-based method", len(defects))
        return defects
    
    def _extract_defect_from_chunk(self, chunk: Dict[str, Any]) -> Optional[DefectInfo]:
        """Extract defect information from a single refined chunk with improved accuracy"""
        description = chunk["description"].lower()
        
        # Word to number mapping for tread numbers and priorities (up to 25)
        word_to_num = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
            'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
            'twenty-one': '21', 'twenty-two': '22', 'twenty-three': '23', 'twenty-four': '24', 'twenty-five': '25',
            'twentyone': '21', 'twentytwo': '22', 'twentythree': '23', 'twentyfour': '24', 'twentyfive': '25'
        }
        
        # Extract tread number - IMPROVED to handle alphabetic numbers (up to 25)
        tread_patterns = [
            # Numeric patterns
            r'(?:tread|track|try|tri|tred|thread)\s+(?:number\s+)?(\d+)',
            r'(?:tread|track|try|tri|tred|thread)\s+(\d+)',
            r'(\d+)\s+(?:tread|track|try|tri|tred|thread)',
            # Alphabetic patterns (up to 25)
            r'(?:tread|track|try|tri|tred|thread)\s+(?:number\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|twentyone|twentytwo|twentythree|twentyfour|twentyfive)',
            r'(?:tread|track|try|tri|tred|thread)\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|twentyone|twentytwo|twentythree|twentyfour|twentyfive)',
            r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|twentyone|twentytwo|twentythree|twentyfour|twentyfive)\s+(?:tread|track|try|tri|tred|thread)'
        ]
        
        tread_number = None
        for pattern in tread_patterns:
            match = re.search(pattern, description)
            if match:
                tread_val = match.group(1)
                # Convert word to number if needed
                tread_number = word_to_num.get(tread_val, tread_val)
                break
        
        # Extract priority - IMPROVED to handle more word variations (up to 25)
        priority_patterns = [
            r'priority\s+(\d+)',
            r'priority\s+(one|two|three|four|five|six)',
            r'(\d+)\s+priority',
            r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|twentyone|twentytwo|twentythree|twentyfour|twentyfive)\s+priority'
        ]
        
        priority = None
        for pattern in priority_patterns:
            match = re.search(pattern, description)
            if match:
                priority_val = match.group(1)
                # Convert word to number
                priority = word_to_num.get(priority_val, priority_val)
                break
        
        # Extract COMPLETE defect description - MAJOR IMPROVEMENT
        defect_description = None
        
        # Strategy 1: Extract complete defect description after priority
        # Look for patterns like "priority X, [complete description]"
        priority_desc_pattern = r'priority\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*,\s*([^,]+?)(?:,|screenshot|$)'
        match = re.search(priority_desc_pattern, description, re.IGNORECASE)
        if match:
            defect_description = match.group(1).strip()
        
        # Strategy 2: Extract complete description between tread and screenshot/end
        if not defect_description:
            tread_desc_pattern = r'(?:tread|track|try|tri|tred|thread).*?priority.*?,\s*([^,]+?)(?:,|screenshot|$)'
            match = re.search(tread_desc_pattern, description, re.IGNORECASE)
            if match:
                defect_description = match.group(1).strip()
        
        # Strategy 3: Look for comprehensive defect patterns
        if not defect_description:
            comprehensive_patterns = [
                # Pattern: "top front and rear cracks"
                r'(top|bottom|front|rear|center)\s+(front|rear|top|bottom|center)?\s*(?:and\s+)?(front|rear|top|bottom|center)?\s*(crack|cracks|defect|defects)',
                # Pattern: "front and rear crack"
                r'(front|rear|top|bottom|center)\s+and\s+(front|rear|top|bottom|center)\s+(crack|cracks|defect|defects)',
                # Pattern: "top, front and rear cracks"
                r'(top|bottom|front|rear|center),\s*(front|rear|top|bottom|center)\s+and\s+(front|rear|top|bottom|center)\s+(crack|cracks|defect|defects)',
                # Simple patterns
                r'(top|bottom|front|rear|center)\s+(crack|cracks|defect|defects)',
                r'(crack|cracks|defect|defects)\s+(top|bottom|front|rear|center)'
            ]
            
            for pattern in comprehensive_patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    defect_description = match.group(0).strip()
                    break
        
        # Strategy 4: Extract everything after the last comma before screenshot/end
        if not defect_description:
            # Find the last comma and extract everything after it (before screenshot/end)
            comma_pattern = r',\s*([^,]+?)(?:,|screenshot|$)'
            matches = re.findall(comma_pattern, description, re.IGNORECASE)
            if matches:
                # Take the last match (most likely the description)
                potential_desc = matches[-1].strip()
                # Check if it contains defect keywords
                if any(keyword in potential_desc for keyword in ['crack', 'defect', 'damage', 'wear', 'broken', 'top', 'bottom', 'front', 'rear']):
                    defect_description = potential_desc
        
        # Strategy 5: Fallback - extract any crack/defect mention with better context
        if not defect_description:
            defect_keywords = ['crack', 'defect', 'damage', 'wear', 'broken']
            for keyword in defect_keywords:
                if keyword in description:
                    # Extract surrounding context (improved)
                    words = description.split()
                    keyword_index = words.index(keyword) if keyword in words else -1
                    if keyword_index >= 0:
                        # Look backwards and forwards for location words
                        start_idx = max(0, keyword_index - 3)
                        end_idx = min(len(words), keyword_index + 4)
                        context_words = words[start_idx:end_idx]
                        
                        # Reconstruct the description
                        defect_description = ' '.join(context_words)
                        break
        
        # Clean up the description
        if defect_description:
            # Remove common trailing words
            defect_description = re.sub(r'\s+(screenshot|screen\s+shot|,)$', '', defect_description, flags=re.IGNORECASE)
            defect_description = defect_description.strip()
        
        # Extract defect if we have any relevant information
        if tread_number or priority or defect_description:
            defect_info = DefectInfo(
                tread_number=tread_number,
                priority=priority,
                description=defect_description,
                timestamp_start=chunk["start_time"],
                timestamp_end=chunk["end_time"],
                transcript_segment=chunk["description"]
            )
            # Calculate screenshot timestamp
            defect_info.ss_timestamp = self._calculate_screenshot_timestamp(chunk["start_time"], chunk["end_time"])
            return defect_info
        
        return None
    
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
    
    def _calculate_screenshot_timestamp(self, start_time: float, end_time: float) -> float:
        """Calculate optimal timestamp for taking screenshot"""
        if start_time is None or end_time is None:
            return None
         
        # Calculate the duration of the timestamp range
        duration = end_time - start_time
        
        # Take screenshot at the start of the 3rd portion (2/3 of the way through)
        # For a 12-second range: 0 to 12 seconds
        # 1st portion: 0-4 seconds (0 to 1/3)
        # 2nd portion: 4-8 seconds (1/3 to 2/3) 
        # 3rd portion: 8-12 seconds (2/3 to end)
        # Screenshot at: 8 seconds (start of 3rd portion)
        screenshot_time = start_time + (duration * 2/3)
        
        return screenshot_time 
    
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
        
        # Save transcript as readable text file for evaluation
        transcript_text_file = os.path.join(output_dir, "transcript.txt")
        self._save_transcript_as_text(transcript_result, transcript_text_file)
        
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
            
            # Read PDF file
            with open(local_pdf_path, 'rb') as f:
                pdf_data = f.read()
            
            # Upload to S3 using presigned URL
            logger.info("📤 Uploading PDF to S3...")
            response = requests.put(presigned_s3_url, data=pdf_data, headers={'Content-Type': 'application/pdf'})
            
            if response.status_code == 200:
                # Extract S3 URL from presigned URL (remove query parameters)
                s3_url = presigned_s3_url.split('?')[0]
                logger.info("✅ PDF uploaded successfully to: %s", s3_url)
                return s3_url
            else:
                logger.error("❌ S3 upload failed with status code: %d", response.status_code)
                logger.error("Response: %s", response.text)
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
            whisper_model_name="base.en",
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
            logger.info("📤 Uploading PDF report to S3...")
            s3_url = processor.upload_pdf_to_s3(local_pdf_path, presigned_s3_url)
            if s3_url:
                logger.info("✅ PDF uploaded to S3 successfully")
                return s3_url
            else:
                logger.error("❌ Failed to upload PDF to S3, returning local path")
                return local_pdf_path
        else:
            logger.info("✅ Processing completed, returning local path")
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
        whisper_model_name="small.en",
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
