import os
import cv2
import base64
import time
import json
import torch
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import streamlit as st
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Configuration
TEMP_DIR = "temp_files"
FRAME_INTERVAL = 1       # Seconds between frames
SEQUENCE_SIZE = 6        # Number of consecutive frames to analyze at once
SEQUENCE_OVERLAP = 2     # Number of frames to overlap between sequences
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"  # Qwen model to use

class QwenFootballAnalyzer:
    def __init__(self, video_path, mode="image", analysis_type="identification"):
        """
        Initialize the football analyzer with Qwen2.5-VL model.
        
        Args:
            video_path: Path to the football match video
            mode: Analysis mode - "image" for frame sequence analysis or "video" for direct video analysis
            analysis_type: Type of analysis - "identification" for event detection or "description" for scene description
        """
        self.video_path = video_path
        self.mode = mode
        self.analysis_type = analysis_type
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_seconds = self.total_frames / self.fps
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create directories for temporary files
        os.makedirs(TEMP_DIR, exist_ok=True)
        os.makedirs(f"{TEMP_DIR}/frames", exist_ok=True)
        if mode == "video":
            os.makedirs(f"{TEMP_DIR}/video_segments", exist_ok=True)
        
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Mode: {mode}")
        print(f"Analysis Type: {analysis_type}")
        print(f"Duration: {self._format_time(self.total_seconds)}")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"FPS: {self.fps:.2f}")
        
        # Initialize Qwen2.5-VL model
        print("Loading Qwen2.5-VL model...")
        
        # Use memory-efficient settings
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",  # Enable flash attention if available
            device_map="auto"
        )
        
        # Set up processor with memory-efficient settings
        # Reduce image resolution to save memory
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            min_pixels=512*28*28,    # Minimum size (smaller than this will be upscaled)
            max_pixels=720*28*28     # Maximum size (larger than this will be downscaled)
        )
        
        # Set global sequence size to a smaller value to save memory
        global SEQUENCE_SIZE
        if self.analysis_type == "description":
            # For description mode, use even fewer frames
            SEQUENCE_SIZE = 3
        
        print("Model loaded successfully!")
    
    def _format_time(self, seconds):
        """Format seconds to MM:SS format"""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def analyze_video(self, progress_callback=None):
        """
        Analyze the entire video based on the selected mode and analysis type
        
        Args:
            progress_callback: Optional callback function to report progress for UI updates
                The callback receives: (segment_idx, total_segments, segment_results, segment_frames)
        """
        print(f"\nPerforming {self.analysis_type} analysis of the entire video using {self.mode} mode...")
        all_results = []
        
        # Divide the video into manageable segments (1-minute chunks) with 10-second overlap
        segment_duration = 60  # 1-minute chunks
        segment_overlap = 10   # 10-second overlap between segments
        
        # Create segments with overlap
        segments = []
        start_time = 0
        while start_time < self.total_seconds:
            end_time = min(start_time + segment_duration, self.total_seconds)
            segments.append((start_time, end_time))
            start_time += segment_duration - segment_overlap
        
        # Create progress bar for segments
        with tqdm(total=len(segments), desc="Segment Analysis") as segment_pbar:
            for segment_idx, (start_time, end_time) in enumerate(segments):
                print(f"\nAnalyzing segment {segment_idx+1}/{len(segments)}: {self._format_time(start_time)} - {self._format_time(end_time)}")
                
                segment_frames = []  # Store frame paths for UI display
                
                if self.mode == "image":
                    # Extract frames for this segment
                    frame_sequences = self._extract_frame_sequences(start_time, end_time)
                    
                    # Collect all frame paths for this segment
                    for seq in frame_sequences:
                        segment_frames.extend(seq['frames'])
                    
                    if self.analysis_type == "description":
                        # Generate descriptions for frame sequences
                        segment_results = self._describe_frame_sequences(frame_sequences, segment_idx, len(segments))
                        print(f"Segment {segment_idx+1}/{len(segments)} complete: Generated {len(segment_results)} descriptions")
                    else:  # identification
                        # Analyze frame sequences for events
                        segment_results = self._analyze_frame_sequences(frame_sequences, segment_idx, len(segments))
                        print(f"Segment {segment_idx+1}/{len(segments)} complete: Found {len(segment_results)} distinct events")
                
                else:  # video mode
                    # Extract video segment
                    video_path = self._extract_video_segment(start_time, end_time)
                    segment_frames = [video_path]  # Store video path for UI
                    
                    if self.analysis_type == "description":
                        # Generate description for video segment
                        segment_results = self._describe_video_segment(video_path, start_time, end_time, segment_idx, len(segments))
                        print(f"Segment {segment_idx+1}/{len(segments)} complete: Generated {len(segment_results)} descriptions")
                    else:  # identification
                        # Analyze video segment for events
                        segment_results = self._analyze_video_segment(video_path, start_time, end_time, segment_idx, len(segments))
                        print(f"Segment {segment_idx+1}/{len(segments)} complete: Found {len(segment_results)} distinct events")
                
                # Call the progress callback with segment results if provided
                if progress_callback:
                    progress_callback(segment_idx, len(segments), segment_results, segment_frames)
                
                all_results.extend(segment_results)
                segment_pbar.update(1)
        
        return all_results

    def _extract_frame_sequences(self, start_time, end_time):
        """Extract frame sequences from the segment"""
        sequences = []
        
        # Calculate frame positions
        start_frame_pos = int(start_time * self.fps)
        end_frame_pos = int(end_time * self.fps)
        
        # Calculate frame step based on desired interval
        frame_step = int(self.fps * FRAME_INTERVAL)
        
        # Extract frames at regular intervals
        frame_paths = []
        timestamps = []
        frame_numbers = []  # Store frame numbers for reference
        
        with tqdm(total=(end_frame_pos - start_frame_pos) // frame_step, desc="Extracting Frames") as pbar:
            for frame_pos in range(start_frame_pos, end_frame_pos, frame_step):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Calculate timestamp
                timestamp = frame_pos / self.fps
                timestamp_str = self._format_time(timestamp)
                
                # Add timestamp and frame number to image
                frame_with_timestamp = frame.copy()
                cv2.putText(
                    frame_with_timestamp,
                    f"Time: {timestamp_str} (Frame: {frame_pos})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Resize the frame to reduce memory usage
                # Calculate target size to maintain aspect ratio while reducing resolution
                target_width = 640  # A smaller width to reduce memory usage
                target_height = int(self.frame_height * (target_width / self.frame_width))
                resized_frame = cv2.resize(frame_with_timestamp, (target_width, target_height))
                
                # Save frame
                frame_filename = f"frame_{int(timestamp):06d}.jpg"
                frame_path = os.path.join(TEMP_DIR, "frames", frame_filename)
                
                # Save with lower JPEG quality to reduce file size
                cv2.imwrite(frame_path, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                frame_paths.append(frame_path)
                timestamps.append(timestamp)
                frame_numbers.append(frame_pos)
                
                pbar.update(1)
        
        # Create sequences with overlap
        for i in range(0, len(frame_paths) - SEQUENCE_SIZE + 1, SEQUENCE_SIZE - SEQUENCE_OVERLAP):
            sequence_frames = frame_paths[i:i+SEQUENCE_SIZE]
            sequence_timestamps = timestamps[i:i+SEQUENCE_SIZE]
            sequence_frame_numbers = frame_numbers[i:i+SEQUENCE_SIZE]
            
            if len(sequence_frames) == SEQUENCE_SIZE:  # Only use complete sequences
                sequences.append({
                    'frames': sequence_frames,
                    'timestamps': sequence_timestamps,
                    'frame_numbers': sequence_frame_numbers,
                    'start_time': sequence_timestamps[0],
                    'end_time': sequence_timestamps[-1]
                })
        
        return sequences
    
    def _extract_video_segment(self, start_time, end_time):
        """Extract a video segment between start_time and end_time"""
        segment_duration = end_time - start_time
        segment_filename = f"segment_{int(start_time):06d}_{int(end_time):06d}.mp4"
        segment_path = os.path.join(TEMP_DIR, "video_segments", segment_filename)
        
        # Check if segment already exists
        if os.path.exists(segment_path):
            return segment_path
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate target size to maintain aspect ratio while reducing resolution
        target_width = 640  # A smaller width to reduce memory usage
        target_height = int(height * (target_width / width))
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            segment_path, 
            fourcc, 
            fps, 
            (target_width, target_height)
        )
        
        # Calculate frame positions
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = end_frame - start_frame
        
        # Extract frames for the segment
        with tqdm(total=total_frames, desc=f"Extracting video segment {self._format_time(start_time)}-{self._format_time(end_time)}") as pbar:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_num in range(start_frame, end_frame):
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Add timestamp and frame number
                timestamp = frame_num / fps
                timestamp_str = self._format_time(timestamp)
                
                cv2.putText(
                    frame,
                    f"Time: {timestamp_str} (Frame: {frame_num})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Resize frame to reduce memory requirements
                resized_frame = cv2.resize(frame, (target_width, target_height))
                
                writer.write(resized_frame)
                pbar.update(1)
        
        writer.release()
        return segment_path

    def _create_analysis_prompt(self, mode="identification"):
        """Create a prompt for analyzing football match content based on the mode"""
        if mode == "description":
            return """You are an expert football match commentator providing detailed play-by-play analysis.

Generate specific, soccer-centric commentary on what's happening in the footage. Focus exclusively on gameplay actions such as:

1. Ball possession and transitions (e.g., "Team A maintaining possession in the midfield")
2. Specific player actions - ALWAYS INCLUDE JERSEY NUMBERS when visible:
   - Passes (successful/unsuccessful, type of pass)
   - Shots (on/off target, blocked)
   - Tackles (successful challenges, interceptions)
   - Dribbles and skills
   - Clearances and defensive actions
3. Goals and scoring opportunities - ALWAYS IDENTIFY WHEN A GOAL OCCURS
   - Be extremely clear and explicit when you see a goal being scored
   - Mention the jersey number of the goal scorer when visible
   - Describe how the goal was scored (header, volley, tap-in, etc.)
4. Set pieces and significant events:
   - Free kicks (who won them, location)
   - Corner kicks (who's taking, defensive setup)
   - Penalties
   - Yellow/red cards (for what actions)
5. Tactical observations:
   - Team formations and positioning
   - Pressing intensity
   - Defensive organization
   - Counter-attacks
   - Build-up play patterns

IMPORTANT:
- ALWAYS refer to players by their jersey numbers when visible (e.g., "Number 10 passes to Number 7")
- ALWAYS mention the exact timestamp or frame number shown in the images when describing key actions
- Be extremely clear when identifying goals, penalties, cards, and other key events
- For EACH significant action, mention the frame number or timestamp where it occurs

Your commentary should read like professional soccer play-by-play analysis, focusing on the specific actions visible in the footage with proper football terminology. Be concise yet detailed about the actual gameplay events occurring.
"""
        else:  # identification mode
            return """You are an expert football analyst identifying key events from football match footage.

Focus ONLY on identifying these five key football events:

1. GOAL
   - Definition: Ball completely crosses goal line between posts and under crossbar
   - Visual indicators:
     * Shot being taken, ball crossing line/entering net
     * Player celebrations with raised arms
     * Goalkeeper retrieving ball from net
     * Referee pointing to center circle
   - ALWAYS note the jersey number of the goal scorer when visible

2. PENALTY
   - Definition: Direct free kick from penalty spot after foul in penalty area
   - Visual indicators:
     * Referee pointing to penalty spot
     * Player positioning on/around penalty spot
     * Goalkeeper on goal line
     * Penalty being taken
   - ALWAYS note the jersey number of the fouled player and the player taking the penalty

3. CORNER KICK
   - Definition: Ball placed at corner arc after crossing goal line off defender
   - Visual indicators:
     * Player at corner flag
     * Players positioning in penalty area
     * Ball being delivered from corner position
   - ALWAYS note the jersey number of the player taking the corner

4. FREE KICK
   - Definition: Set piece after a foul
   - Visual indicators:
     * Defensive wall formation
     * Player(s) standing over the ball
     * Referee positioning defenders at proper distance
   - ALWAYS note the jersey number of the player taking the free kick

5. CARD (YELLOW/RED)
   - Definition: Disciplinary action by referee
   - Visual indicators:
     * Referee showing card to specific player
     * Player reaction to card
   - ALWAYS note the jersey number of the player receiving the card

IMPORTANT GUIDELINES:
- CAREFULLY VERIFY that a GOAL has actually been scored - look for the ball crossing the line, celebrations, etc.
- Pay careful attention to jersey numbers and always include them in your descriptions
- ONLY identify these 5 specific events - no other football actions
- Look at the chronological sequence to understand the flow of play
- Provide the EXACT time range AND frame numbers for the event based on the visible timestamps
- Be very specific about what visual evidence you see
- If you're uncertain about an event, DO NOT include it

Return your findings in this EXACT format:
[MM:SS-MM:SS] (Frames: XXX-YYY) EVENT_TYPE: Brief description with specific visual evidence and player jersey numbers

Example:
[35:20-35:28] (Frames: 4224-4236) GOAL: Number 9 strikes ball from outside box, enters net past goalkeeper's dive, followed by celebration with teammates. Ball clearly crosses goal line at frame 4230.

If no clear events are detected, respond with "No key events detected in this sequence."
"""

    def _describe_frame_sequences(self, sequences, segment_idx, total_segments):
        """Generate descriptions for frame sequences using Qwen2.5-VL"""
        if not sequences:
            return []
        
        segment_descriptions = []
        
        # Create progress bar for sequences
        with tqdm(total=len(sequences), desc="Sequence Description") as seq_pbar:
            for seq_idx, sequence in enumerate(sequences):
                frames = sequence['frames']
                timestamps = sequence['timestamps']
                frame_numbers = sequence['frame_numbers']
                start_time = sequence['start_time']
                end_time = sequence['end_time']
                
                start_time_str = self._format_time(start_time)
                end_time_str = self._format_time(end_time)
                
                # Create a single message with text and all images
                user_content = [
                    {
                        "type": "text",
                        "text": f"""Describe what's happening in these {len(frames)} consecutive frames from a football match. 
Time range: {start_time_str} to {end_time_str}
Frame numbers: {frame_numbers[0]} to {frame_numbers[-1]}
The frames are in chronological order.

IMPORTANT: 
1. Always mention jersey numbers when visible (e.g., "Number 10 passes to Number 7")
2. Identify any goals, penalties, free kicks, corner kicks, or cards that occur
3. For each key action, mention the specific frame number or timestamp where it occurs
"""
                    }
                ]
                
                # Add all images to the content array
                for frame_path in frames:
                    user_content.append({
                        "type": "image",
                        "image": f"{frame_path}"
                    })
                
                messages = [
                    {
                        "role": "system",
                        "content": self._create_analysis_prompt(mode="description")
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ]
                
                # Define frame_ids for reference in results
                frame_ids = [os.path.basename(frame) for frame in frames]
                
                # Qwen2.5-VL inference
                try:
                    # Clear CUDA cache to free up memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Process messages
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(self.model.device)
                    
                    # Generate response with memory-efficient settings
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs, 
                            max_new_tokens=1000,  # Set to 1000 as requested
                            do_sample=False      # Deterministic generation saves memory
                        )
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_text = self.processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                    
                    # Log the raw output for debugging
                    print(f"\nRaw model output for sequence {seq_idx+1}:")
                    print(output_text)
                    
                    
                    # Add description to results
                    description_string = f"[{start_time_str}-{end_time_str}] (Frames: {frame_numbers[0]}-{frame_numbers[-1]}) DESCRIPTION: {output_text}"
                    
                    segment_descriptions.append({
                        'description_string': description_string,
                        'frame_ids': frame_ids,
                        'start_time': start_time,
                        'end_time': end_time,
                        'frame_numbers': frame_numbers,
                        'description': output_text,
                        'goal_detected': goal_mentioned
                    })
                    
                    print(f"  Generated description for sequence {seq_idx+1}/{len(sequences)}")
                    
                except Exception as e:
                    print(f"Error describing sequence {start_time_str}-{end_time_str}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(2)  # Wait before retry
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                seq_pbar.update(1)
        
        print(f"Segment {segment_idx+1}/{total_segments} complete: Generated {len(segment_descriptions)} descriptions")
        
        return segment_descriptions

    def _analyze_frame_sequences(self, sequences, segment_idx, total_segments):
        """Analyze frame sequences using Qwen2.5-VL"""
        if not sequences:
            return []
        
        segment_events = []
        
        # Create progress bar for sequences
        with tqdm(total=len(sequences), desc="Sequence Analysis") as seq_pbar:
            for seq_idx, sequence in enumerate(sequences):
                frames = sequence['frames']
                timestamps = sequence['timestamps']
                frame_numbers = sequence['frame_numbers']
                start_time = sequence['start_time']
                end_time = sequence['end_time']
                
                start_time_str = self._format_time(start_time)
                end_time_str = self._format_time(end_time)
                
                # Create a single message with text and all images in the content array
                user_content = [
                    {
                        "type": "text",
                        "text": f"""
Analyze these {len(frames)} consecutive frames from a football match.

Time range: {start_time_str} to {end_time_str}
Frame numbers: {frame_numbers[0]} to {frame_numbers[-1]}

The frames are in chronological order.
Each frame has its timestamp and frame number displayed.

Identify if any of these SPECIFIC football events are occurring:
- GOAL
- PENALTY
- CORNER KICK
- FREE KICK
- CARD (YELLOW/RED)

ALWAYS include jersey numbers of involved players when visible.
ONLY report events you're absolutely certain about with precise visual evidence.
"""
                    }
                ]
                
                # Add all frames to the content array
                for frame_path in frames:
                    user_content.append({
                        "type": "image",
                        "image": f"{frame_path}"
                    })
                
                # Create messages for Qwen2.5-VL
                messages = [
                    {
                        "role": "system",
                        "content": self._create_analysis_prompt(mode="identification")
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ]
                
                # Define frame_ids for reference in results
                frame_ids = [os.path.basename(frame) for frame in frames]
                
                # Qwen2.5-VL inference
                try:
                    # Clear CUDA cache to free up memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Process messages
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(self.model.device)
                    
                    # Generate response with memory-efficient settings
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs, 
                            max_new_tokens=1000,  # Set to 1000 as requested
                            do_sample=False      # Deterministic generation saves memory
                        )
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_text = self.processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                    
                    # Log the raw output for debugging
                    print(f"\nRaw model output for sequence {seq_idx+1}:")
                    print(output_text[:200] + "..." if len(output_text) > 200 else output_text)
                    
                    # Parse events with time ranges
                    events = self._parse_events(output_text, start_time, end_time, frame_ids, frame_numbers)
                    segment_events.extend(events)
                    
                    # Print events found in this sequence
                    if events:
                        print(f"  Sequence {seq_idx+1}/{len(sequences)}: Found {len(events)} events")
                        for event in events:
                            print(f"    [{self._format_time(event['start_time'])}-{self._format_time(event['end_time'])}] (Frames: {event['frame_range'][0]}-{event['frame_range'][1]}) {event['event_type']}")
                            print(f"    Frames: {', '.join(event['frame_ids'])}")
                    else:
                        print(f"  Sequence {seq_idx+1}/{len(sequences)}: No events detected")
                    
                except Exception as e:
                    print(f"Error analyzing sequence {start_time_str}-{end_time_str}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(2)  # Wait before retry
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                seq_pbar.update(1)
        
        # Post-process: merge similar events that are close in time
        merged_events = self._merge_events(segment_events)
        
        print(f"Segment {segment_idx+1}/{total_segments} complete: Found {len(merged_events)} distinct events")
        
        return merged_events
        
    def _describe_video_segment(self, video_path, start_time, end_time, segment_idx, total_segments):
        """Generate description for a video segment using Qwen2.5-VL"""
        segment_start_str = self._format_time(start_time)
        segment_end_str = self._format_time(end_time)
        
        # Calculate approximate frame range
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        # Create messages for Qwen2.5-VL with proper content structure
        messages = [
            {
                "role": "system",
                "content": self._create_analysis_prompt(mode="description")
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
Describe what's happening in this football match video segment.

Time range: {segment_start_str} to {segment_end_str}
Frame range: {start_frame} to {end_frame}

The video has timestamps and frame numbers displayed.
Provide a detailed description of the football match action, focusing on:
- Player positions and movements WITH JERSEY NUMBERS (e.g., "Number 10 passes to Number 7")
- Ball possession and play development
- Team formations and tactics
- Notable player actions (passes, dribbles, tackles, etc.)
- Field position where the action is taking place
- ALWAYS identify any goals, penalties, free kicks, corner kicks, or cards that occur
- For each key action, mention the specific timestamp or frame number where it occurs
"""
                    },
                    {
                        "type": "video",
                        "video": f"{video_path}",
                        "fps": 1.0,  # Process 1 frame per second
                        "max_pixels": 720*28*28  # Limit resolution for memory efficiency
                    }
                ]
            }
        ]
        
        print(f"Generating description for video segment: {segment_start_str}-{segment_end_str}")
        
        try:
            # Clear CUDA cache to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Process messages
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                fps=1.0,  # Manually pass fps
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response with memory-efficient settings
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=1000,  # Set to 1000 as requested
                    do_sample=False      # Deterministic generation saves memory
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            
            # Log the raw output for debugging
            print(f"\nRaw model output for video segment description:")
            print(output_text)
            
            # Check for goal mentions in description mode
            goal_mentioned = False
            if "goal" in output_text.lower() or "scored" in output_text.lower():
                goal_mentioned = True
                print(f"GOAL DETECTED in description mode at {segment_start_str}-{segment_end_str}")
            
            # Create description result
            description_string = f"[{segment_start_str}-{segment_end_str}] (Frames: {start_frame}-{end_frame}) DESCRIPTION: {output_text}"
            
            description_result = [{
                'description_string': description_string,
                'frame_ids': [os.path.basename(video_path)],
                'start_time': start_time,
                'end_time': end_time,
                'frame_numbers': [start_frame, end_frame],
                'description': output_text,
                'goal_detected': goal_mentioned
            }]
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return description_result
            
        except Exception as e:
            print(f"Error describing video segment {segment_start_str}-{segment_end_str}: {str(e)}")
            import traceback
            traceback.print_exc()
            time.sleep(2)  # Wait before retry
            return []
    
    def _analyze_video_segment(self, video_path, start_time, end_time, segment_idx, total_segments):
        """Analyze a video segment using Qwen2.5-VL"""
        segment_start_str = self._format_time(start_time)
        segment_end_str = self._format_time(end_time)
        
        # Calculate approximate frame range
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        # Create messages for Qwen2.5-VL with proper content structure
        messages = [
            {
                "role": "system",
                "content": self._create_analysis_prompt(mode="identification")
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
Analyze this video segment from a football match.

Time range: {segment_start_str} to {segment_end_str}
Frame range: {start_frame} to {end_frame}

The video has timestamps and frame numbers displayed.

Identify if any of these SPECIFIC football events are occurring:
- GOAL
- PENALTY
- CORNER KICK
- FREE KICK
- CARD (YELLOW/RED)

ALWAYS include jersey numbers of involved players when visible.
Pay extra attention to detect any goals that may occur during the sequence.
For each event, note the specific timestamp or frame number where it occurs.
ONLY report events you're absolutely certain about with precise visual evidence.
"""
                    },
                    {
                        "type": "video",
                        "video": f"{video_path}",
                        "fps": 1.0,  # Process 1 frame per second
                        "max_pixels": 720*28*28  # Limit resolution for memory efficiency
                    }
                ]
            }
        ]
        
        print(f"Analyzing video segment: {segment_start_str}-{segment_end_str}")
        
        try:
            # Clear CUDA cache to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Process messages
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                fps=1.0,  # Manually pass fps
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response with memory-efficient settings
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=1000,  # Set to 1000 as requested
                    do_sample=False      # Deterministic generation saves memory
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            
            # Log the raw output for debugging
            print(f"\nRaw model output for video segment:")
            print(output_text[:200] + "..." if len(output_text) > 200 else output_text)
            
            # Parse events with time ranges
            events = self._parse_events(output_text, start_time, end_time, [os.path.basename(video_path)], [start_frame, end_frame])
            
            # Post-process: merge similar events that are close in time
            merged_events = self._merge_events(events)
            
            print(f"Segment {segment_idx+1}/{total_segments} complete: Found {len(merged_events)} distinct events")
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return merged_events
            
        except Exception as e:
            print(f"Error analyzing video segment {segment_start_str}-{segment_end_str}: {str(e)}")
            import traceback
            traceback.print_exc()
            time.sleep(2)  # Wait before retry
            return []

    def _parse_events(self, event_text, window_start_time, window_end_time, frame_ids, frame_numbers=None):
        """Parse events from model response"""
        events = []
        
        if not event_text or "No key events detected" in event_text:
            return events
            
        lines = event_text.strip().split('\n')
        
        for line in lines:
            if '[' in line and ']' in line and ':' in line:
                # Extract time range
                time_range = line[line.find('[')+1:line.find(']')]
                
                # Initialize frame range
                frame_range = None
                
                # Check for frame information
                if "Frames:" in line or "Frame:" in line:
                    try:
                        frame_info = line[line.find("Frames:")+7:] if "Frames:" in line else line[line.find("Frame:")+6:]
                        frame_info = frame_info.split(")")[0] if ")" in frame_info else frame_info.split(" ")[0]
                        
                        if "-" in frame_info:
                            start_frame, end_frame = map(int, frame_info.split("-"))
                            frame_range = [start_frame, end_frame]
                        else:
                            # Single frame mentioned
                            try:
                                single_frame = int(frame_info.strip())
                                frame_range = [single_frame, single_frame]
                            except ValueError:
                                pass
                    except:
                        # If parsing fails, use the provided frame numbers if available
                        if frame_numbers and len(frame_numbers) >= 2:
                            frame_range = [min(frame_numbers), max(frame_numbers)]
                
                # If still no frame range, use the provided frame numbers if available
                if not frame_range and frame_numbers:
                    if isinstance(frame_numbers, list) and len(frame_numbers) >= 2:
                        frame_range = [min(frame_numbers), max(frame_numbers)]
                    else:
                        # Default to None if we can't determine frame range
                        frame_range = None
                
                if '-' in time_range:
                    # Already has a time range
                    start_str, end_str = time_range.split('-')
                    
                    # Convert to seconds for internal processing
                    try:
                        start_minutes, start_seconds = map(int, start_str.split(':'))
                        end_minutes, end_seconds = map(int, end_str.split(':'))
                        
                        start_time = start_minutes * 60 + start_seconds
                        end_time = end_minutes * 60 + end_seconds
                    except ValueError:
                        # If time format is invalid, use window times
                        start_time = window_start_time
                        end_time = window_end_time
                else:
                    # Single time point, create a small range
                    try:
                        minutes, seconds = map(int, time_range.split(':'))
                        point_time = minutes * 60 + seconds
                        
                        # Create a 3-second window around the point
                        start_time = point_time - 1.5
                        end_time = point_time + 1.5
                    except ValueError:
                        # If time format is invalid, use window times
                        start_time = window_start_time
                        end_time = window_end_time
                
                # Extract event type and description
                remaining = line[line.find(']')+1:].strip()
                if ':' in remaining:
                    event_type_part, description = remaining.split(':', 1)
                    
                    # Handle cases where frame info might be part of event_type_part
                    if "Frames:" in event_type_part or "Frame:" in event_type_part:
                        event_type_part = event_type_part.split("(")[0].strip()
                    
                    event_type = event_type_part.strip().upper()  # Normalize event type
                    description = description.strip()
                    
                    # Only accept the 5 specific event types we're looking for
                    if any(et in event_type for et in ["GOAL", "PENALTY", "CORNER", "FREE KICK", "CARD"]):
                        # Standardize event types
                        if "GOAL" in event_type:
                            event_type = "GOAL"
                        elif "PENALTY" in event_type:
                            event_type = "PENALTY"
                        elif "CORNER" in event_type:
                            event_type = "CORNER KICK"
                        elif "FREE" in event_type:
                            event_type = "FREE KICK"
                        elif "CARD" in event_type:
                            event_type = "CARD"
                            
                        events.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'event_type': event_type,
                            'description': description,
                            'frame_ids': frame_ids,
                            'frame_range': frame_range
                        })
        
        return events

    def _merge_events(self, events):
        """Merge similar events that are close in time"""
        if not events:
            return []
        
        # Group events by type
        events_by_type = defaultdict(list)
        for event in events:
            events_by_type[event['event_type']].append(event)
        
        merged_events = []
        
        # Process each event type
        for event_type, type_events in events_by_type.items():
            # Sort by start time
            type_events.sort(key=lambda x: x['start_time'])
            
            # Merge overlapping events
            current_group = [type_events[0]]
            
            for event in type_events[1:]:
                last_event = current_group[-1]
                
                # Check if events overlap or are very close (within 3 seconds)
                if event['start_time'] <= last_event['end_time'] + 3:
                    # Merge events
                    last_event['end_time'] = max(last_event['end_time'], event['end_time'])
                    # Keep the more detailed description
                    if len(event['description']) > len(last_event['description']):
                        last_event['description'] = event['description']
                    # Keep track of all frames that captured this event
                    if 'all_frame_ids' not in last_event:
                        last_event['all_frame_ids'] = last_event['frame_ids']
                    last_event['all_frame_ids'].extend(event['frame_ids'])
                    
                    # Update frame range if available
                    if 'frame_range' in event and 'frame_range' in last_event:
                        if event['frame_range'] and last_event['frame_range']:
                            last_event['frame_range'] = [
                                min(last_event['frame_range'][0], event['frame_range'][0]),
                                max(last_event['frame_range'][1], event['frame_range'][1])
                            ]
                else:
                    # Add frame_ids list to the event being added to the group
                    event['all_frame_ids'] = event['frame_ids']
                    current_group.append(event)
            
            merged_events.extend(current_group)
        
        # Convert back to formatted strings
        formatted_events = []
        for event in merged_events:
            start_str = self._format_time(event['start_time'])
            end_str = self._format_time(event['end_time'])
            all_frame_ids = event.get('all_frame_ids', event['frame_ids'])
            # Remove duplicates while preserving order
            unique_frames = []
            for frame in all_frame_ids:
                if frame not in unique_frames:
                    unique_frames.append(frame)
            
            # Format with frame range if available
            if 'frame_range' in event and event['frame_range']:
                event_string = f"[{start_str}-{end_str}] (Frames: {event['frame_range'][0]}-{event['frame_range'][1]}) {event['event_type']}: {event['description']}"
            else:
                event_string = f"[{start_str}-{end_str}] {event['event_type']}: {event['description']}"
            
            formatted_events.append({
                'event_string': event_string,
                'frame_ids': unique_frames,
                'start_time': event['start_time'],  # Keep this for sorting
                'end_time': event['end_time'],
                'event_type': event['event_type'],
                'description': event['description'],
                'frame_range': event.get('frame_range')
            })
        
        # Sort by start time
        formatted_events.sort(key=lambda x: x['start_time'])
        
        return formatted_events

    def _cleanup(self, keep_frames=False):
        """Clean up temporary files, optionally keeping frames"""
        print("\nCleaning up temporary files...")
        self.cap.release()
        
        if os.path.exists(TEMP_DIR) and not keep_frames:
            frames_dir = f"{TEMP_DIR}/frames"
            if os.path.exists(frames_dir):
                for file in os.listdir(frames_dir):
                    try:
                        os.remove(os.path.join(frames_dir, file))
                    except:
                        pass
                try:
                    os.rmdir(frames_dir)
                except:
                    pass
            
            videos_dir = f"{TEMP_DIR}/video_segments"
            if os.path.exists(videos_dir):
                for file in os.listdir(videos_dir):
                    try:
                        os.remove(os.path.join(videos_dir, file))
                    except:
                        pass
                try:
                    os.rmdir(videos_dir)
                except:
                    pass
            
            # Remove any other files in the temp directory
            for file in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
            
            # Only remove TEMP_DIR if not keeping frames
            if not keep_frames:
                try:
                    os.rmdir(TEMP_DIR)
                except:
                    pass

# Main function to use the analyzer
def analyze_football_video(video_path, output_path=None, mode="image", analysis_type="identification", progress_callback=None):
    """
    Analyze football video and save results
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the results (default: based on video filename)
        mode: Analysis mode - "image" or "video" (default: "image")
        analysis_type: Type of analysis - "identification" or "description" (default: "identification")
        progress_callback: Optional callback function for UI progress updates
    """
    if not output_path:
        output_path = os.path.splitext(video_path)[0] + f"_{mode}_{analysis_type}_results.txt"
    
    analyzer = QwenFootballAnalyzer(video_path, mode=mode, analysis_type=analysis_type)
    # Call analyze_video with progress callback
    results = analyzer.analyze_video(progress_callback=progress_callback)
    
    # Clean up temporary files but keep frames
    analyzer._cleanup(keep_frames=True)
    
    # Save results
    with open(output_path, 'w') as f:
        if analysis_type == "description":
            f.write("FOOTBALL MATCH DESCRIPTIONS\n")
            f.write("===========================\n\n")
            f.write(f"Analysis mode: {mode}\n\n")
            for result in results:
                f.write(f"{result['description_string']}\n")
                f.write(f"  Frames: {', '.join(result['frame_ids'])}\n\n")
            
            print(f"\nResults saved to {output_path}")
            print("\nGenerated Descriptions:")
            for result in results:
                time_range = result['description_string'].split(']')[0].strip('[')
                print(f"[{time_range}] Description")
                print(result['description'][:100] + "..." if len(result['description']) > 100 else result['description'])
                print()
        else:  # identification
            f.write("FOOTBALL MATCH KEY EVENTS\n")
            f.write("==========================\n\n")
            f.write(f"Analysis mode: {mode}\n\n")
            for result in results:
                f.write(f"{result['event_string']}\n")
                f.write(f"  Frames: {', '.join(result['frame_ids'])}\n\n")
            
            print(f"\nResults saved to {output_path}")
            print("\nDetected Events:")
            for result in results:
                print(result['event_string'])
                print(f"  Frames: {', '.join(result['frame_ids'])}")
                print()
    
    return results

# Enhanced Streamlit app function with live updates
def streamlit_app():
    st.title("Football Match Analysis with Qwen2.5-VL")
    st.write("Analyze football matches to detect events or generate descriptions")
    
    # Add styling for the streaming mode
    st.markdown("""
    <style>
    .highlight-segment {
        border: 2px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(76, 175, 80, 0.1);
        margin-bottom: 10px;
    }
    .detected-event {
        border-left: 4px solid #FF5722;
        padding-left: 10px;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Analysis mode selection
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.radio(
            "Select processing mode:",
            ["image", "video"],
            index=0,
            help="""
            - Image mode: Analyzes sequences of frames
            - Video mode: Processes video segments directly
            """
        )
    
    with col2:
        analysis_type = st.radio(
            "Select analysis type:",
            ["identification", "description"],
            index=0,
            help="""
            - Identification: Detects key events (goals, penalties, corners, free kicks, cards)
            - Description: Generates detailed descriptions of what's happening in the match
            """
        )
    
    uploaded_file = st.file_uploader("Upload a football match video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        col1, col2 = st.columns(2)
        with col1:
            streaming_mode = st.checkbox("Enable streaming mode (live updates)", value=True)
        with col2:
            analyze_button = st.button("Start Analysis", type="primary")
        
        # Initialize placeholders for streaming UI
        progress_container = st.empty()
        segment_status = st.empty()
        current_segment_container = st.empty()
        all_results_container = st.empty()
        
        # Create tabs for results
        if streaming_mode:
            tab1, tab2 = st.tabs(["Current Segment", "All Results"])
            
            with tab1:
                current_segment_ui = st.empty()
            
            with tab2:
                all_results_ui = st.empty()
                all_events_list = []
        
        if analyze_button:
            if not streaming_mode:
                progress_bar = progress_container.progress(0)
                status_text = segment_status.text("Starting analysis...")
                
                try:
                    # Run analysis without streaming
                    status_text = segment_status.text(f"Analyzing video with {mode} mode, {analysis_type} analysis...")
                    progress_bar.progress(30)
                    
                    output_file = f"match_{mode}_{analysis_type}_results.txt"
                    results = analyze_football_video("temp_video.mp4", output_file, mode=mode, analysis_type=analysis_type)
                    
                    progress_bar.progress(100)
                    status_text = segment_status.text("Analysis complete!")
                    
                    # Display results all at once
                    if analysis_type == "description":
                        with all_results_container.container():
                            st.success(f"Generated {len(results)} descriptions")
                            
                            st.subheader("Football Match Descriptions")
                            for i, result in enumerate(results):
                                time_range = result['description_string'].split(']')[0].strip('[')
                                with st.expander(f"Description for {time_range}"):
                                    st.write(result['description'])
                                    
                                    # Display frames for this description (up to 4)
                                    frames_to_show = result['frame_ids'][:4]  # Limit to 4 frames/videos
                                    
                                    if mode == "image" and all(frame.startswith("frame_") for frame in frames_to_show):
                                        # These are image frames
                                        cols = st.columns(min(2, len(frames_to_show)))  # Up to 2 columns
                                        
                                        for j, frame_id in enumerate(frames_to_show):
                                            frame_path = os.path.join(TEMP_DIR, "frames", frame_id)
                                            if os.path.exists(frame_path):
                                                cols[j % 2].image(frame_path, caption=f"Frame {frame_id}")
                                    else:
                                        # These are video segments
                                        st.write("Video segment:")
                                        for segment_id in frames_to_show:
                                            st.write(f"- {segment_id}")
                    else:
                        with all_results_container.container():
                            st.success(f"Detected {len(results)} key events")
                            
                            st.subheader("Key Football Events")
                            for i, result in enumerate(results):
                                with st.expander(f"{result['event_string']}"):
                                    st.write(result['event_string'])
                                    
                                    # Display frames for this event (up to 4)
                                    frames_to_show = result['frame_ids'][:4]  # Limit to 4 frames/videos
                                    
                                    if mode == "image" and all(frame.startswith("frame_") for frame in frames_to_show):
                                        # These are image frames
                                        cols = st.columns(min(2, len(frames_to_show)))  # Up to 2 columns
                                        
                                        for j, frame_id in enumerate(frames_to_show):
                                            frame_path = os.path.join(TEMP_DIR, "frames", frame_id)
                                            if os.path.exists(frame_path):
                                                cols[j % 2].image(frame_path, caption=f"Frame {frame_id}")
                                    else:
                                        # These are video segments
                                        st.write("Video segment:")
                                        for segment_id in frames_to_show:
                                            st.write(f"- {segment_id}")
                    
                    with open(output_file, "r") as f:
                        st.download_button(
                            label="Download Results",
                            data=f.read(),
                            file_name=f"football_{mode}_{analysis_type}_results.txt",
                            mime="text/plain"
                        )
                except Exception as e:
                    segment_status.error(f"Error during analysis: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
            
            else:
                # Streaming mode with live updates
                progress_bar = progress_container.progress(0)
                status_text = segment_status.text("Starting analysis...")
                
                # Define callback for progress updates
                def update_progress(segment_idx, total_segments, segment_results, segment_frames):
                    # Update progress bar
                    progress = int((segment_idx + 1) / total_segments * 100)
                    progress_bar.progress(progress)
                    
                    # Update status text
                    status_text.text(f"Analyzing segment {segment_idx + 1}/{total_segments} ({progress}% complete)")
                    
                    # Update current segment display
                    with current_segment_ui.container():
                        st.markdown(f"### Current segment: {segment_idx + 1}/{total_segments}")
                        
                        # Show sample frames from current segment
                        if segment_frames:
                            if mode == "image":
                                # Show up to 4 frames
                                sample_frames = segment_frames[:4]
                                cols = st.columns(min(2, len(sample_frames)))
                                
                                for i, frame in enumerate(sample_frames):
                                    if os.path.exists(frame):
                                        cols[i % 2].image(frame, caption=f"Frame {os.path.basename(frame)}")
                            else:  # video mode
                                if os.path.exists(segment_frames[0]):
                                    st.video(segment_frames[0])
                        
                        # Show segment results
                        if analysis_type == "description":
                            for result in segment_results:
                                st.markdown(f"**{result['description_string'].split('DESCRIPTION:')[0]}**")
                                st.write(result['description'])
                                
                                # Highlight if a goal was detected
                                if result.get('goal_detected', False):
                                    st.success("⚽ GOAL DETECTED in this segment!")
                        else:  # identification
                            if segment_results:
                                st.markdown(f"**Found {len(segment_results)} events in this segment:**")
                                for result in segment_results:
                                    st.markdown(f"**{result['event_type']}** {result['event_string'].split(':', 1)[1] if ':' in result['event_string'] else ''}")
                                    
                                    # For goals, show the frames
                                    if result['event_type'] == "GOAL":
                                        st.success("⚽ GOAL DETECTED!")
                                        if mode == "image":
                                            # Find the corresponding frames 
                                            event_frames = result['frame_ids'][:2]  # Show up to 2 frames
                                            cols = st.columns(min(2, len(event_frames)))
                                            for i, frame_id in enumerate(event_frames):
                                                frame_path = os.path.join(TEMP_DIR, "frames", frame_id)
                                                if os.path.exists(frame_path):
                                                    cols[i].image(frame_path, caption=f"Frame {frame_id}")
                            else:
                                st.write("No events detected in this segment.")
                    
                    # Add to cumulative results
                    nonlocal all_events_list
                    all_events_list.extend(segment_results)
                    
                    # Update cumulative results display
                    with all_results_ui.container():
                        st.markdown(f"### All Results ({len(all_events_list)} total)")
                        
                        if analysis_type == "description":
                            # Group by time ranges for descriptions
                            for i, result in enumerate(all_events_list):
                                time_range = result['description_string'].split(']')[0].strip('[')
                                with st.expander(f"Description for {time_range}"):
                                    st.write(result['description'])
                                    
                                    # Flag if goal was detected
                                    if result.get('goal_detected', False):
                                        st.success("⚽ GOAL DETECTED in this segment!")
                                    
                                    # Display frames for this description
                                    if mode == "image" and len(result['frame_ids']) > 0:
                                        # These are image frames - show the first one
                                        frame_id = result['frame_ids'][0]
                                        frame_path = os.path.join(TEMP_DIR, "frames", frame_id)
                                        if os.path.exists(frame_path):
                                            st.image(frame_path, caption=f"Sample Frame")
                        else:  # identification
                            # Group by event types
                            event_types = set(event['event_type'] for event in all_events_list)
                            
                            for event_type in event_types:
                                type_events = [event for event in all_events_list if event['event_type'] == event_type]
                                st.markdown(f"#### {event_type} Events ({len(type_events)})")
                                
                                for event in type_events:
                                    with st.expander(f"{event['event_string'].split(event_type + ':')[0]}{event_type}"):
                                        st.write(event['description'] if 'description' in event else 
                                                 event['event_string'].split(':', 1)[1] if ':' in event['event_string'] else '')
                                        
                                        # Display frames for this event
                                        if mode == "image" and len(event['frame_ids']) > 0:
                                            # For goals, show more frames
                                            max_frames = 4 if event_type == "GOAL" else 2
                                            frames_to_show = event['frame_ids'][:max_frames]
                                            cols = st.columns(min(2, len(frames_to_show)))
                                            
                                            for j, frame_id in enumerate(frames_to_show):
                                                frame_path = os.path.join(TEMP_DIR, "frames", frame_id)
                                                if os.path.exists(frame_path):
                                                    cols[j % 2].image(frame_path, caption=f"Frame {frame_id}")
                                        elif mode == "video" and len(event['frame_ids']) > 0:
                                            # For video mode, try to show the video segment
                                            video_path = os.path.join(TEMP_DIR, "video_segments", event['frame_ids'][0])
                                            if os.path.exists(video_path):
                                                st.video(video_path)

                try:
                    # Initialize the results list
                    all_events_list = []
                    
                    # Run analysis with streaming callback
                    output_file = f"match_{mode}_{analysis_type}_results.txt"
                    results = analyze_football_video("temp_video.mp4", output_file, mode=mode, analysis_type=analysis_type, 
                                                    progress_callback=update_progress)
                    
                    # Analysis complete
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    # Final summary and download button
                    st.success(f"Analysis complete! Found {len(results)} total {analysis_type} entries.")
                    
                    with open(output_file, "r") as f:
                        st.download_button(
                            label="Download Results",
                            data=f.read(),
                            file_name=f"football_{mode}_{analysis_type}_results.txt",
                            mime="text/plain"
                        )
                except Exception as e:
                    segment_status.error(f"Error during analysis: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
            
            # Clean up
            if os.path.exists("temp_video.mp4"):
                os.remove("temp_video.mp4")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Football Match Analysis with Qwen2.5-VL")
    parser.add_argument("--video", type=str, help="Path to football match video")
    parser.add_argument("--output", type=str, help="Path to output file (optional)")
    parser.add_argument("--mode", type=str, 
                        choices=["image", "video"], 
                        default="image",
                        help="Processing mode: 'image' for frame sequences, 'video' for direct video processing")
    parser.add_argument("--analysis", type=str,
                        choices=["identification", "description"],
                        default="identification",
                        help="Analysis type: 'identification' for event detection, 'description' for match descriptions")
    
    args = parser.parse_args()
    
    if args.video:
        analyze_football_video(args.video, args.output, mode=args.mode, analysis_type=args.analysis)
    else:
        # If no command line args, try to run as Streamlit app
        try:
            streamlit_app()
        except Exception as e:
            print(f"Error running Streamlit app: {str(e)}")
            print("To run as CLI: python script.py --video path/to/video.mp4 --mode [image|video] --analysis [identification|description]")
            print("To run as Streamlit app: streamlit run script.py")