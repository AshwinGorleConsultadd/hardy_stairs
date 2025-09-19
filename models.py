"""
Pydantic Models for Video Processing Pipeline

This module contains all the Pydantic models used throughout the application.
"""

from typing import Optional
from pydantic import BaseModel, Field


class DefectInfo(BaseModel):
    """Pydantic model for structured defect information"""
    building_counter: Optional[str] = Field(None, description="Building counter (building1, building2, etc.)")
    building_name: Optional[str] = Field(None, description="Building name if mentioned in video")
    apartment_number: Optional[str] = Field(None, description="Apartment number")
    tread_number: Optional[str] = Field(None, description="Tread number mentioned")
    priority: Optional[str] = Field(None, description="Priority level (1, 2, etc.) or (one, two, etc.)")
    description: Optional[str] = Field(None, description="Description of the defect")
    timestamp_start: Optional[float] = Field(None, description="Start timestamp in seconds")
    timestamp_end: Optional[float] = Field(None, description="End timestamp in seconds")
    ss_timestamp: Optional[float] = Field(None, description="Screenshot timestamp in seconds")
    transcript_segment: Optional[str] = Field(None, description="Original transcript segment")
