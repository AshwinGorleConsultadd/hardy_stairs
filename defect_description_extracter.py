"""
Defect Description Extractor Module

This module contains functions for extracting defect information from transcript chunks.
"""

import re
from typing import Optional, Dict, Any
from models import DefectInfo


def extract_defect_from_chunk(chunk: Dict[str, Any]) -> Optional[DefectInfo]:
    """Extract defect information from a single refined chunk with improved accuracy"""
    # Safety check for None or invalid chunk
    if chunk is None or not isinstance(chunk, dict):
        return None
    
    if "description" not in chunk or not chunk["description"]:
        return None
    
    description = chunk["description"].lower()
    
    # Word to number mapping for tread numbers and priorities (up to 25)
    word_to_num = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'twenty-one': '21', 'twenty-two': '22', 'twenty-three': '23', 'twenty-four': '24', 'twenty-five': '25'
    }
    
    # Initialize variables
    tread_number = None
    priority = None
    defect_description = None
    
    # Strategy 1: Extract tread number (numeric and alphabetic)
    tread_patterns = [
        r'tread\s+(?:number\s+)?(\d+)',
        r'track\s+(?:number\s+)?(\d+)',
        r'try\s+(?:number\s+)?(\d+)',
        r'tri\s+(?:number\s+)?(\d+)',
        r'tred\s+(?:number\s+)?(\d+)',
        r'thread\s+(?:number\s+)?(\d+)',
        r'tread\s+(?:number\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five)',
        r'track\s+(?:number\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five)',
        r'try\s+(?:number\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five)',
        r'tri\s+(?:number\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five)',
        r'tred\s+(?:number\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five)',
        r'thread\s+(?:number\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five)'
    ]
    
    for pattern in tread_patterns:
        match = re.search(pattern, description)
        if match:
            tread_value = match.group(1)
            if tread_value in word_to_num:
                tread_number = word_to_num[tread_value]
            else:
                tread_number = tread_value
            break
    
    # Strategy 2: Extract priority (numeric and alphabetic)
    priority_patterns = [
        r'priority\s+(\d+)',
        r'priority\s+(one|two|three|four|five|six)',
        r'pri\s+(\d+)',
        r'pri\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five)'
    ]
    
    for pattern in priority_patterns:
        match = re.search(pattern, description)
        if match:
            priority_value = match.group(1)
            if priority_value in word_to_num:
                priority = word_to_num[priority_value]
            else:
                priority = priority_value
            break
    
    # Strategy 3: Extract defect description using multiple approaches
    defect_description = extract_defect_description(description)
    
    # Strategy 4: If no description found, try to extract from the entire chunk
    if not defect_description:
        defect_description = extract_defect_description_from_full_chunk(chunk["description"])
    
    # Strategy 5: Clean up the description
    if defect_description:
        defect_description = clean_defect_description(defect_description)
    
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
        return defect_info
    
    return None


def extract_defect_description(description: str) -> Optional[str]:
    """Extract defect description from the text"""
    # Common defect patterns
    defect_patterns = [
        r'(top\s+(?:front\s+)?rear\s+crack)',
        r'(bottom\s+(?:front\s+)?rear\s+crack)',
        r'(top\s+(?:rear\s+)?front\s+crack)',
        r'(bottom\s+(?:rear\s+)?front\s+crack)',
        r'(top\s+center\s+crack)',
        r'(bottom\s+center\s+crack)',
        r'(top\s+rear\s+crack)',
        r'(bottom\s+rear\s+crack)',
        r'(top\s+front\s+crack)',
        r'(bottom\s+front\s+crack)',
        r'(rear\s+crack)',
        r'(front\s+crack)',
        r'(top\s+crack)',
        r'(bottom\s+crack)',
        r'(center\s+crack)',
        r'(crack)'
    ]
    
    for pattern in defect_patterns:
        match = re.search(pattern, description)
        if match:
            return match.group(1)
    
    return None


def extract_defect_description_from_full_chunk(full_text: str) -> Optional[str]:
    """Extract defect description from the full chunk text"""
    # Look for crack-related terms
    crack_patterns = [
        r'(top\s+(?:front\s+)?rear\s+crack)',
        r'(bottom\s+(?:front\s+)?rear\s+crack)',
        r'(top\s+(?:rear\s+)?front\s+crack)',
        r'(bottom\s+(?:rear\s+)?front\s+crack)',
        r'(top\s+center\s+crack)',
        r'(bottom\s+center\s+crack)',
        r'(top\s+rear\s+crack)',
        r'(bottom\s+rear\s+crack)',
        r'(top\s+front\s+crack)',
        r'(bottom\s+front\s+crack)',
        r'(rear\s+crack)',
        r'(front\s+crack)',
        r'(top\s+crack)',
        r'(bottom\s+crack)',
        r'(center\s+crack)',
        r'(crack)'
    ]
    
    for pattern in crack_patterns:
        match = re.search(pattern, full_text.lower())
        if match:
            return match.group(1)
    
    return None


def clean_defect_description(description: str) -> str:
    """Clean up the defect description"""
    if not description:
        return description
    
    # Remove common prefixes
    description = re.sub(r'^(priority\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|\d+)\s*)', '', description, flags=re.IGNORECASE)
    description = re.sub(r'^(tread\s+(?:number\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|\d+)\s*)', '', description, flags=re.IGNORECASE)
    description = re.sub(r'^(track\s+(?:number\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|\d+)\s*)', '', description, flags=re.IGNORECASE)
    description = re.sub(r'^(try\s+(?:number\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|\d+)\s*)', '', description, flags=re.IGNORECASE)
    description = re.sub(r'^(tri\s+(?:number\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|\d+)\s*)', '', description, flags=re.IGNORECASE)
    description = re.sub(r'^(tred\s+(?:number\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|\d+)\s*)', '', description, flags=re.IGNORECASE)
    description = re.sub(r'^(thread\s+(?:number\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|\d+)\s*)', '', description, flags=re.IGNORECASE)
    
    # Remove common suffixes
    description = re.sub(r'\s+(screenshot|screen\s+shot|,)$', '', description, flags=re.IGNORECASE)
    description = description.strip()
    
    return description


def calculate_screenshot_timestamp(start_time: float, end_time: float) -> float:
    """Calculate screenshot timestamp (2/3 through the duration)"""
    if start_time is None or end_time is None:
        return None
    
    try:
        duration = end_time - start_time
        screenshot_time = start_time + (duration * 2/3)
        return screenshot_time
    except (TypeError, ValueError) as e:
        print(f"Error calculating screenshot timestamp: {e}")
        return None
