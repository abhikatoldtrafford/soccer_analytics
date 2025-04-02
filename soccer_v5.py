import os
import cv2
import base64
import time
from datetime import datetime
import json
import numpy as np
import tempfile
from datetime import timedelta
from tqdm import tqdm
from openai import OpenAI
import streamlit as st
from collections import defaultdict, Counter, deque
import threading
from PIL import Image
import io
import traceback
import torch
import pandas as pd
import logging
from pathlib import Path
import uuid
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Import components from detect_track.py
from detect_track_v3 import SoccerTracker, PlayerReIDTracker, JerseyOCR, extract_player_highlights

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("soccer_analytics.log")
    ]
)
logger = logging.getLogger("SoccerAnalytics")

# Initialize OpenAI client with Streamlit secrets
if 'OPENAI_API_KEY' in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    # Fallback to environment variable if not running in Streamlit
    import os
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
WINDOW_SIZE = 15         # Seconds per analysis window (reduced from 30 to 15)
WINDOW_STEP = 12         # Seconds to advance window between analyses
FRAME_INTERVAL = 1       # Seconds between frames (increased to capture fewer frames)
SEQUENCE_SIZE = 10       # Number of consecutive frames to analyze at once (reduced from 30)
MODEL = "gpt-4o-mini"    # Model to use for analysis
RESIZE_DIMENSION = 800   # Resize image dimension for API (smaller size = fewer tokens)
TRACK_INTERVAL = 3       # Process every nth frame for tracking (to improve performance)

# Debug mode - set to True to print more information
DEBUG = True
import subprocess
import pandas as pd
import hashlib

# Function to get a unique identifier for a video file
def load_events_from_file(events_file_path):
    """Load and format events from a JSON file"""
    if not os.path.exists(events_file_path):
        logger.error(f"Events file not found: {events_file_path}")
        return []
    
    try:
        with open(events_file_path, 'r') as f:
            data = json.load(f)
        
        # Verify and format events if needed
        formatted_events = []
        for event in data:
            # Ensure required fields exist
            if 'start_time' in event and 'end_time' in event and 'event_type' in event:
                # Create event_string if missing
                if 'event_string' not in event:
                    start_str = f"{int(event['start_time'] // 60):02d}:{int(event['start_time'] % 60):02d}"
                    end_str = f"{int(event['end_time'] // 60):02d}:{int(event['end_time'] % 60):02d}"
                    event['event_string'] = f"[{start_str}-{end_str}] {event['event_type']}"
                
                # Ensure frame_ids exists
                if 'frame_ids' not in event:
                    event['frame_ids'] = []
                
                formatted_events.append(event)
        
        logger.info(f"Loaded {len(formatted_events)} events from {events_file_path}")
        return formatted_events
    
    except Exception as e:
        logger.error(f"Error loading events from {events_file_path}: {str(e)}")
        if DEBUG:
            traceback.print_exc()
        return []
def get_video_hash(video_path):
    """Generate a unique hash for a video file based on filename and size"""
    if not os.path.exists(video_path):
        return None
    
    file_size = os.path.getsize(video_path)
    file_name = os.path.basename(video_path)
    
    # Create hash based on filename and size
    hash_input = f"{file_name}_{file_size}"
    video_hash = hashlib.md5(hash_input.encode()).hexdigest()
    
    return video_hash

# Function to check if analysis exists for a video
def check_existing_analysis(video_path):
    """Check if analysis already exists for this video in our database"""
    analysis_db_path = "analysis_database.csv"
    
    if not os.path.exists(analysis_db_path):
        # Create the database if it doesn't exist
        columns = ['video_hash', 'video_path', 'analysis_timestamp', 'output_directory', 
                   'events_json_path', 'tracking_complete', 'highlights_generated']
        analysis_db = pd.DataFrame(columns=columns)
        analysis_db.to_csv(analysis_db_path, index=False)
        return None
    
    # Load the database
    try:
        analysis_db = pd.read_csv(analysis_db_path)
        
        # Get hash for current video
        video_hash = get_video_hash(video_path)
        if video_hash is None:
            return None
        
        # Check if this video exists in our database
        matching_entries = analysis_db[analysis_db['video_hash'] == video_hash]
        
        if len(matching_entries) == 0:
            return None
            
        # Return the most recent analysis
        latest_entry = matching_entries.sort_values('analysis_timestamp', ascending=False).iloc[0]
        
        # Verify that the output directory exists
        if not os.path.exists(latest_entry['output_directory']):
            logger.warning(f"Output directory {latest_entry['output_directory']} not found")
            return None
        for key, value in latest_entry.items():
            # Replace NaN values with empty string for string fields
            if isinstance(value, float) and pd.isna(value) and key in ['events_json_path', 'output_directory', 'video_path']:
                latest_entry[key] = ''
        return latest_entry.to_dict()
    
    except Exception as e:
        logger.error(f"Error checking analysis database: {str(e)}")
        return None

# Function to save analysis metadata to database
def save_analysis_metadata(video_path, output_directory, events_json_path=None, tracking_complete=False, highlights_generated=False):
    """Save analysis metadata to the database"""
    analysis_db_path = "analysis_database.csv"
    
    try:
        # Create or load the database
        if os.path.exists(analysis_db_path):
            analysis_db = pd.read_csv(analysis_db_path)
        else:
            columns = ['video_hash', 'video_path', 'analysis_timestamp', 'output_directory', 
                     'events_json_path', 'tracking_complete', 'highlights_generated']
            analysis_db = pd.DataFrame(columns=columns)
        
        # Get hash for current video
        video_hash = get_video_hash(video_path)
        if video_hash is None:
            logger.error(f"Could not generate hash for {video_path}")
            return False
        
        # Check if this video exists in our database
        matching_entries = analysis_db[analysis_db['video_hash'] == video_hash]
        
        # Prepare new entry
        new_entry = {
            'video_hash': video_hash,
            'video_path': video_path,
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'output_directory': output_directory,
            'events_json_path': events_json_path if events_json_path else '',
            'tracking_complete': tracking_complete,
            'highlights_generated': highlights_generated
        }
        
        # If entry exists, update it; otherwise add new entry
        if len(matching_entries) > 0:
            # Remove all existing entries for this video
            analysis_db = analysis_db[analysis_db['video_hash'] != video_hash]
        
        # Add the new entry
        analysis_db = pd.concat([analysis_db, pd.DataFrame([new_entry])], ignore_index=True)
        
        # Save the database
        analysis_db.to_csv(analysis_db_path, index=False)
        logger.info(f"Saved analysis metadata for {video_path} to database")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to analysis database: {str(e)}")
        if DEBUG:
            traceback.print_exc()
        return False

# Function to run detect_track.py as a subprocess
def run_tracking_pipeline(video_path, output_dir=None, yolo_model="yolov8n.pt", 
                         reid_model="osnet_x1_0", reid_weights="osnet_x1_0_market1501.pth", 
                         device="cuda"):
    """Run the tracking pipeline directly using imported classes"""
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False, "Video file not found", None
    
    # Create output directory if not provided
    if output_dir is None:
        video_hash = get_video_hash(video_path)
        output_dir = f"output_{video_hash}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify GPU is available if requested
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    elif device == "cuda" and torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_info}")
    else:
        logger.info("Using CPU for processing")
    
    try:
        # Create tracker instance
        logger.info(f"Initializing SoccerTracker with models: {yolo_model}, {reid_model}")
        tracker = SoccerTracker(
            yolo_model_path=yolo_model,
            reid_model_name=reid_model,
            reid_weights_path=reid_weights,
            device=device,
            ocr_interval=10,  # Every 10 frames run OCR
            conf=0.3,  # Detection confidence
            reid_threshold=0.8,
            max_history=900
        )
        
        # Create an output video path
        output_video_path = os.path.join(output_dir, "tracked_video.mp4")
        
        # Process the video
        logger.info(f"Processing video: {video_path}")
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return False, "Failed to open video", None
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        
        # Process frames
        logger.info(f"Processing {total_frames} frames at {fps} FPS")
        
        # Set frame interval for processing (to improve performance)
        frame_interval = 3  # Process every 3rd frame
        
        for frame_idx in range(0, total_frames, frame_interval):
            # Show progress every 50 frames
            if frame_idx % 50 == 0:
                progress = frame_idx / total_frames
                logger.info(f"Processing frame {frame_idx}/{total_frames} ({progress:.1%})")
            
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}")
                break
                
            # Process frame with tracker
            annotated_frame, tracks = tracker.process_frame(frame)
            
            # Write frame
            writer.write(annotated_frame)
            
        # Release resources
        cap.release()
        writer.release()
        
        logger.info(f"Video processing complete. Output saved to: {output_video_path}")
        
        # Generate player highlight data
        highlight_data_path = os.path.join(output_dir, "highlight_data.json")
        tracker.generate_player_highlight_data(
            output_path=highlight_data_path,
            min_segment_duration=1.0,
            min_total_duration=3.0
        )
        
        # Generate trajectory visualization for all players
        trajectory_path = os.path.join(output_dir, "player_trajectories.jpg")
        tracker.generate_trajectory(trajectory_path, frame_size=(w, h))
        
        # Extract player highlights
        extract_player_highlights(
            video_path=video_path,
            highlight_data_path=highlight_data_path,
            output_dir=os.path.join(output_dir, "highlight_clips"),
            max_clips=5,
            highlight_seconds=2.0
        )
        
        logger.info(f"All processing complete. Results saved to: {output_dir}")
        return True, "Processing complete", output_dir
        
    except Exception as e:
        logger.error(f"Error in tracking pipeline: {str(e)}")
        if DEBUG:
            traceback.print_exc()
        return False, str(e), None

# Function to check tracking process status
def check_tracking_status(process):
    """Check if tracking process is still running and read output"""
    if process.poll() is None:
        # Process still running
        return True, "Running", process.stdout.readline() if process.stdout else ""
    else:
        # Process completed
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            return False, "Completed", stdout
        else:
            return False, "Failed", stderr
# Add the two functions at the beginning of the file, right after imports
def find_highlight_video_for_player(player_key, output_dir=None):
    """Find existing highlight video for a player across all directories"""
    # Handle both GID and jersey number formats
    jersey_prefix = f"player_{player_key}" if isinstance(player_key, int) else None
    gid_prefix = None if isinstance(player_key, int) else f"player_GID{player_key.replace('gid_', '')}"
    
    # Check all possible paths
    possible_paths = []
    
    # Add output directory paths if provided
    if output_dir:
        if jersey_prefix:
            possible_paths.extend([
                os.path.join(output_dir, "highlight_clips", f"{jersey_prefix}_highlights.mp4"),
                os.path.join(output_dir, f"{jersey_prefix}_highlights.mp4"),
                os.path.join(output_dir, "highlight_clips", f"{jersey_prefix}_clip_1.mp4"),
                os.path.join(output_dir, f"{jersey_prefix}_clip_1.mp4")
            ])
        if gid_prefix:
            possible_paths.extend([
                os.path.join(output_dir, "highlight_clips", f"{gid_prefix}_highlights.mp4"),
                os.path.join(output_dir, f"{gid_prefix}_highlights.mp4"),
                os.path.join(output_dir, "highlight_clips", f"{gid_prefix}_clip_1.mp4"),
                os.path.join(output_dir, f"{gid_prefix}_clip_1.mp4")
            ])
    
    # Add jersey number paths if applicable
    if jersey_prefix:
        possible_paths.extend([
            f"output/highlight_clips/{jersey_prefix}_highlights.mp4",
            f"highlights/videos/{jersey_prefix}_highlights.mp4", 
            f"player_highlights/{jersey_prefix}_highlights.mp4",
            f"output/highlight_clips/{jersey_prefix}_clip_1.mp4",
            f"highlights/videos/{jersey_prefix}_clip_1.mp4"
        ])
    
    # Add GID paths if applicable
    if gid_prefix:
        possible_paths.extend([
            f"output/highlight_clips/{gid_prefix}_highlights.mp4",
            f"highlights/videos/{gid_prefix}_highlights.mp4",
            f"player_highlights/{gid_prefix}_highlights.mp4",
            f"output/highlight_clips/{gid_prefix}_clip_1.mp4",
            f"highlights/videos/{gid_prefix}_clip_1.mp4"
        ])
    
    # Check all paths
    for path in possible_paths:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            logger.info(f"Found highlight video for player {player_key}: {path}")
            return path
    
    return None

def scan_existing_highlight_videos(output_dir=None):
    """Scan for existing highlight videos and return a mapping of player IDs to video paths"""
    highlight_videos = {}
    
    # Check both possible directories for highlights
    highlight_dirs = ["highlights/videos", "output/highlight_clips"]
    
    # Add output directory if provided
    if output_dir:
        highlight_dirs.extend([
            os.path.join(output_dir, "highlight_clips"),
            output_dir
        ])
    
    for directory in highlight_dirs:
        if not os.path.exists(directory):
            continue
            
        # Scan for highlight videos
        for filename in os.listdir(directory):
            if filename.endswith(".mp4"):
                # Try to extract player ID from filename
                # Common patterns:
                # - player_GID176_highlights.mp4 (for global IDs)
                # - player_50_highlights.mp4 (for jersey numbers)
                if filename.startswith("player_"):
                    video_path = os.path.join(directory, filename)
                    parts = filename.split("_")
                    
                    if len(parts) >= 2:
                        # Check for GID pattern
                        if "GID" in parts[1]:
                            # Extract GID number from string like "GID176"
                            gid_num = parts[1].replace("GID", "")
                            if gid_num.isdigit():
                                player_id = f"gid_{gid_num}"
                                highlight_videos[player_id] = video_path
                        
                        # Check for jersey number pattern
                        elif parts[1].isdigit():
                            player_id = int(parts[1])
                            highlight_videos[player_id] = video_path
    
    logger.info(f"Found {len(highlight_videos)} existing highlight videos")
    return highlight_videos

class FootballPlayerTracker:
    """
    Tracks players during detected key events using SoccerTracker
    """
    def __init__(self, video_path, yolo_model_path="yolov8n.pt", 
                 reid_model_name="osnet_x1_0", reid_weights_path="osnet_x1_0_market1501.pth",
                 device=None):
        """Initialize the player tracker"""
        self.video_path = video_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create temp directory for track outputs
        self.temp_dir = tempfile.mkdempdir()
        
        # Add output directory property - this will be set from analysis metadata
        self.output_dir = None
        
        # Storage for tracked players in events
        self.event_player_tracks = {}
        self.player_thumbnails = {}  # global_id -> thumbnail image path
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap.release()
        
        # Initialize the player tracker with error handling
        try:
            # Check if model files exist first
            if not os.path.exists(yolo_model_path):
                logger.error(f"YOLO model file not found: {yolo_model_path}")
                raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")
                
            if not os.path.exists(reid_weights_path):
                logger.error(f"ReID weights file not found: {reid_weights_path}")
                raise FileNotFoundError(f"ReID weights not found: {reid_weights_path}")
                
            logger.info(f"Initializing SoccerTracker for player identification using device: {self.device}")
            self.tracker = SoccerTracker(
                yolo_model_path=yolo_model_path,
                reid_model_name=reid_model_name,
                reid_weights_path=reid_weights_path,
                device=self.device,
                ocr_interval=10  # Every 10 frames run OCR to detect jersey numbers
            )
            logger.info("SoccerTracker initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SoccerTracker: {str(e)}")
            # Create a fallback tracker or handle the error
            self.tracker = None
            raise
    
    def _format_time(self, seconds):
        """Format seconds to MM:SS format"""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def analyze_video(self, progress_callback=None, event_callback=None):
        """Analyze the entire video using a rolling window approach"""
        print("\nPerforming analysis of the entire video...")
        all_events = []
        
        # Create sliding windows
        windows = []
        for start_time in range(0, int(self.total_seconds) - WINDOW_SIZE + 1, WINDOW_STEP):
            end_time = start_time + WINDOW_SIZE
            windows.append((start_time, end_time))
        
        total_windows = len(windows)
        
        # Process each window
        for window_idx, (start_time, end_time) in enumerate(windows):
            print(f"\nAnalyzing window {window_idx+1}/{len(windows)}: {self._format_time(start_time)} - {self._format_time(end_time)}")
            
            # Update progress callback
            if progress_callback:
                progress_callback(window_idx / total_windows)
            
            # Extract frames for this window
            frames_data = self._extract_frames(start_time, end_time)
            
            # Analyze frames
            window_events = self._analyze_frames(
                frames_data, 
                window_idx, 
                total_windows,
                event_callback=event_callback
            )
            
            with self.events_lock:
                all_events.extend(window_events)
                self.events_cache = all_events.copy()
        
        # Final progress update
        if progress_callback:
            progress_callback(1.0)
        
        return all_events

    def _extract_frames(self, start_time, end_time):
        """Extract frames from a time window"""
        # Calculate frame positions
        start_frame_pos = int(start_time * self.fps)
        end_frame_pos = int(end_time * self.fps)
        
        # Calculate frame step based on desired interval
        frame_step = int(self.fps * FRAME_INTERVAL)
        
        # Extract frames at regular intervals
        frames_data = []
        
        with tqdm(total=(end_frame_pos - start_frame_pos) // frame_step, desc="Extracting Frames") as pbar:
            for frame_pos in range(start_frame_pos, end_frame_pos, frame_step):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Calculate timestamp
                timestamp = frame_pos / self.fps
                timestamp_str = self._format_time(timestamp)
                
                # Resize the frame to optimize API calls (maintain aspect ratio)
                h, w = frame.shape[:2]
                scale = min(RESIZE_DIMENSION / w, RESIZE_DIMENSION / h)
                new_size = (int(w * scale), int(h * scale))
                resized_frame = cv2.resize(frame, new_size)
                
                # Add timestamp to image
                frame_with_timestamp = resized_frame.copy()
                cv2.putText(
                    frame_with_timestamp,
                    f"Time: {timestamp_str}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Encode directly to base64
                _, buffer = cv2.imencode(".jpg", frame_with_timestamp)
                base64_frame = base64.b64encode(buffer).decode("utf-8")
                
                # Generate a unique ID for this frame
                frame_id = f"frame_{int(timestamp):06d}"
                
                # Store in memory
                frames_data.append({
                    'base64': base64_frame,
                    'timestamp_str': timestamp_str,
                    'frame_id': frame_id,
                    'timestamp': timestamp
                })
                
                # Save frame to disk for display in Streamlit
                frame_path = os.path.join(self.temp_dir, f"{frame_id}.jpg")
                cv2.imwrite(frame_path, frame)
                
                pbar.update(1)
        
        return frames_data

    def _create_player_analysis_prompt(self):
        """Create an improved prompt for analyzing player performance from frames"""
        return """You are an expert football event classifier. Your task is to analyze these frames from a football/soccer match and identify the MOST SIGNIFICANT event occurring.

    ONLY classify the sequence into ONE of these categories:

    1. GOAL
    • Ball crossing the goal line completely between the goalposts
    • Players celebrating with arms raised
    • Goalkeeper retrieving ball from inside the net
    • Scoreboard change or celebration immediately after a shot
    • Teams returning to half-line positions for kickoff

    2. SHOT
    • Player striking the ball toward goal with clear shooting technique
    • Ball traveling toward goal area
    • Goalkeeper diving or attempting a save
    • Ball hitting the post/crossbar
    • Defender making emergency block of shot attempt

    3. SAVE
    • Goalkeeper making contact with the ball to prevent a goal
    • Goalkeeper diving, jumping or extending arms to stop a shot
    • Goalkeeper catching, parrying or deflecting the ball
    • Goalkeeper on the ground with ball after a save
    • Defenders reacting to goalkeeper save

    4. GOOD_PASS
    • Player executing a precise pass that creates attacking advantage
    • Long-range switch of play that changes attack direction
    • Ball moving between players with clear intent
    • Receiving player gaining significant advantage from pass
    • Players positioned strategically for passing sequence

    5. THROUGH_BALL
    • Ball passing between defenders into space
    • Pass breaking defensive line or structure
    • Player running onto ball behind defense
    • Ball directed into space ahead of attacking player
    • Defenders turning toward their own goal to chase the pass

    6. SKILL_MOVE
    • Player performing dribble that beats opponent(s)
    • Technical move like step-over, feint, or roulette
    • Player maintaining close ball control under pressure
    • Defender beaten by skill move (off-balance or wrong-footed)
    • Creative ball manipulation to create space or advantage

    7. NONE
    • Normal play without significant events
    • Players positioned without clear attacking or defending action
    • Paused play, substitutions, or ordinary ball movement
    • Players walking or jogging without urgency

    Analyze all frames collectively as a sequence. If multiple events occur, select the MOST significant one (GOAL > SHOT > SAVE > THROUGH_BALL > GOOD_PASS > SKILL_MOVE).

    Return ONLY the event type without any explanation or description (e.g., "GOAL", "SHOT", "SAVE", "GOOD_PASS", "THROUGH_BALL", "SKILL_MOVE", or "NONE").
    """
    def _analyze_frames(self, frames_data, window_idx, total_windows, event_callback=None):
        """Analyze frames using OpenAI API with optimized token usage"""
        window_events = []
        
        # Skip if not enough frames
        if len(frames_data) < SEQUENCE_SIZE:
            return window_events
        
        # Get first and last timestamp for the window
        start_time = frames_data[0]['timestamp']
        end_time = frames_data[-1]['timestamp']
        start_time_str = self._format_time(start_time)
        end_time_str = self._format_time(end_time)
        
        # Create simplified message content with minimal text
        message_content = [
            {"type": "text", "text": "Classify what significant football action is happening in these frames:"}
        ]
        
        # Add fewer frames to reduce token usage
        # Select frames evenly distributed across the window
        indices = np.linspace(0, len(frames_data) - 1, min(SEQUENCE_SIZE, len(frames_data)), dtype=int)
        selected_frames = [frames_data[i] for i in indices]
        
        # Use medium detail instead of high to reduce tokens
        for frame_data in selected_frames:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_data['base64']}",
                    "detail": "high"  # Changed from high to low to reduce token usage
                }
            })
        
        # Define frame_ids for reference in results
        frame_ids = [frame_data['frame_id'] for frame_data in frames_data]
        
        # API call with max tokens limit to prevent long responses
        try:
            if DEBUG:
                print(f"Analyzing window {window_idx+1}/{total_windows}: {start_time_str} to {end_time_str} with {len(selected_frames)} frames")
            
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": self._create_player_analysis_prompt()},
                    {"role": "user", "content": message_content}
                ],
                temperature=0.2,
            )
            
            event_text = response.choices[0].message.content.strip()
            
            if DEBUG:
                print(f"API Response: {event_text}")
            
            # Skip if no event detected
            if event_text == "NONE" or "No significant actions" in event_text:
                return window_events
                
            # Create a simplified event
            event = {
                'start_time': start_time,
                'end_time': end_time,
                'event_type': event_text.strip(),
                'description': f"Event detected between {start_time_str} and {end_time_str}",
                'frame_ids': frame_ids
            }
            
            # Add event to window_events
            window_events.append(event)
            
            # Call event_callback for live updates if provided
            if event_callback and window_events:
                if DEBUG:
                    print(f"Calling event_callback with event: {event['event_type']}")
                event_callback([event])
            
            # Print event found in this window
            if window_events:
                print(f"  Window {window_idx+1}/{total_windows}: Found event {event['event_type']}")
            
        except Exception as e:
            print(f"Error analyzing window {start_time_str}-{end_time_str}: {str(e)}")
            if DEBUG:
                traceback.print_exc()
            time.sleep(2)  # Wait before trying next window
        
        return window_events

    def _parse_events(self, event_text, window_start_time, window_end_time, frame_ids):
        """Parse events from OpenAI response - simplified to handle single event type classification"""
        events = []
        
        # Skip processing if no meaningful event detected
        if not event_text or "NONE" in event_text or "No significant actions" in event_text:
            return events
            
        # Clean up the response
        event_text = event_text.strip().upper()
        
        # Map of valid event types and their normalized versions
        valid_event_map = {
            "GOAL": "GOAL",
            "GOOD_PASS": "GOOD PASS",
            "THROUGH_BALL": "THROUGH BALL",
            "SHOT": "SHOT ON TARGET",
            "SAVE": "GOOD GOALKEEPING",
            "SKILL_MOVE": "SKILL MOVE",
            "GOOD SAVE": "GOOD GOALKEEPING",
            "GOALKEEPER": "GOOD GOALKEEPING"
        }
        
        # Find matching event type
        matched_event = None
        for key, value in valid_event_map.items():
            if key in event_text:
                matched_event = value
                break
        
        # If no match found but we have some text, use it as is
        if not matched_event and len(event_text) > 0 and len(event_text) < 30:
            matched_event = event_text
        
        # Create event if we found a valid type
        if matched_event:
            events.append({
                'start_time': window_start_time,
                'end_time': window_end_time,
                'event_type': matched_event,
                'description': f"Event detected at {self._format_time(window_start_time)}",
                'frame_ids': frame_ids
            })
        
        return events

    def _merge_events(self, events):
        """Simplified merging of similar events that are close in time"""
        if not events:
            return []
        
        # Already simplified events, so minimal merging needed
        formatted_events = []
        
        # Process each event directly
        for event in events:
            start_str = self._format_time(event['start_time'])
            end_str = self._format_time(event['end_time'])
            
            # Format and add event
            formatted_events.append({
                'event_string': f"[{start_str}-{end_str}] {event['event_type']}",
                'frame_ids': event['frame_ids'],
                'start_time': event['start_time'],
                'end_time': event['end_time'],
                'event_type': event['event_type'],
                'description': f"Event detected between {start_str} and {end_str}"
            })
        
        # Sort by start time
        formatted_events.sort(key=lambda x: x['start_time'])
        
        return formatted_events

    def analyze(self, progress_callback=None, event_callback=None):
        """Main analysis pipeline with callbacks for live updates"""
        try:
            # Analyze the video
            raw_events = self.analyze_video(progress_callback, event_callback)
            
            # Format the events (this is the missing step that adds 'event_string')
            formatted_events = self._merge_events(raw_events)
            
            return formatted_events
                
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            if DEBUG:
                traceback.print_exc()
            return []
    
    def get_frame_path(self, frame_id):
        """Get path to a specific frame"""
        return os.path.join(self.temp_dir, f"{frame_id}.jpg")
    
    def cleanup(self):
        """Clean up temporary files"""
        print("\nCleaning up temporary files...")
        self.cap.release()
        
        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, file))
                except:
                    pass
            try:
                os.rmdir(self.temp_dir)
            except:
                pass


class FootballPlayerAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_seconds = self.total_frames / self.fps
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Event cache for live updates
        self.events_cache = []
        self.events_lock = threading.Lock()
        
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Duration: {self._format_time(self.total_seconds)}")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"FPS: {self.fps:.2f}")
    
    def _format_time(self, seconds):
        """Format seconds to MM:SS format"""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def analyze_video(self, progress_callback=None, event_callback=None):
        """Analyze the entire video using a rolling window approach"""
        print("\nPerforming analysis of the entire video...")
        all_events = []
        
        # Create sliding windows
        windows = []
        for start_time in range(0, int(self.total_seconds) - WINDOW_SIZE + 1, WINDOW_STEP):
            end_time = start_time + WINDOW_SIZE
            windows.append((start_time, end_time))
        
        total_windows = len(windows)
        
        # Process each window
        for window_idx, (start_time, end_time) in enumerate(windows):
            print(f"\nAnalyzing window {window_idx+1}/{len(windows)}: {self._format_time(start_time)} - {self._format_time(end_time)}")
            
            # Update progress callback
            if progress_callback:
                progress_callback(window_idx / total_windows)
            
            # Extract frames for this window
            frames_data = self._extract_frames(start_time, end_time)
            
            # Analyze frames
            window_events = self._analyze_frames(
                frames_data, 
                window_idx, 
                total_windows,
                event_callback=event_callback
            )
            
            with self.events_lock:
                all_events.extend(window_events)
                self.events_cache = all_events.copy()
        
        # Final progress update
        if progress_callback:
            progress_callback(1.0)
        
        return all_events

    def _extract_frames(self, start_time, end_time):
        """Extract frames from a time window"""
        # Calculate frame positions
        start_frame_pos = int(start_time * self.fps)
        end_frame_pos = int(end_time * self.fps)
        
        # Calculate frame step based on desired interval
        frame_step = int(self.fps * FRAME_INTERVAL)
        
        # Extract frames at regular intervals
        frames_data = []
        
        with tqdm(total=(end_frame_pos - start_frame_pos) // frame_step, desc="Extracting Frames") as pbar:
            for frame_pos in range(start_frame_pos, end_frame_pos, frame_step):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Calculate timestamp
                timestamp = frame_pos / self.fps
                timestamp_str = self._format_time(timestamp)
                
                # Resize the frame to optimize API calls (maintain aspect ratio)
                h, w = frame.shape[:2]
                scale = min(RESIZE_DIMENSION / w, RESIZE_DIMENSION / h)
                new_size = (int(w * scale), int(h * scale))
                resized_frame = cv2.resize(frame, new_size)
                
                # Add timestamp to image
                frame_with_timestamp = resized_frame.copy()
                cv2.putText(
                    frame_with_timestamp,
                    f"Time: {timestamp_str}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Encode directly to base64
                _, buffer = cv2.imencode(".jpg", frame_with_timestamp)
                base64_frame = base64.b64encode(buffer).decode("utf-8")
                
                # Generate a unique ID for this frame
                frame_id = f"frame_{int(timestamp):06d}"
                
                # Store in memory
                frames_data.append({
                    'base64': base64_frame,
                    'timestamp_str': timestamp_str,
                    'frame_id': frame_id,
                    'timestamp': timestamp
                })
                
                # Save frame to disk for display in Streamlit
                frame_path = os.path.join(self.temp_dir, f"{frame_id}.jpg")
                cv2.imwrite(frame_path, frame)
                
                pbar.update(1)
        
        return frames_data

    def _create_player_analysis_prompt(self):
        """Create an improved prompt for analyzing player performance from frames"""
        return """You are an expert football event classifier. Your task is to analyze these frames from a football/soccer match and identify the MOST SIGNIFICANT event occurring.

    ONLY classify the sequence into ONE of these categories:

    1. GOAL
    • Ball crossing the goal line completely between the goalposts
    • Players celebrating with arms raised
    • Goalkeeper retrieving ball from inside the net
    • Scoreboard change or celebration immediately after a shot
    • Teams returning to half-line positions for kickoff

    2. SHOT
    • Player striking the ball toward goal with clear shooting technique
    • Ball traveling toward goal area
    • Goalkeeper diving or attempting a save
    • Ball hitting the post/crossbar
    • Defender making emergency block of shot attempt

    3. SAVE
    • Goalkeeper making contact with the ball to prevent a goal
    • Goalkeeper diving, jumping or extending arms to stop a shot
    • Goalkeeper catching, parrying or deflecting the ball
    • Goalkeeper on the ground with ball after a save
    • Defenders reacting to goalkeeper save

    4. GOOD_PASS
    • Player executing a precise pass that creates attacking advantage
    • Long-range switch of play that changes attack direction
    • Ball moving between players with clear intent
    • Receiving player gaining significant advantage from pass
    • Players positioned strategically for passing sequence

    5. THROUGH_BALL
    • Ball passing between defenders into space
    • Pass breaking defensive line or structure
    • Player running onto ball behind defense
    • Ball directed into space ahead of attacking player
    • Defenders turning toward their own goal to chase the pass

    6. SKILL_MOVE
    • Player performing dribble that beats opponent(s)
    • Technical move like step-over, feint, or roulette
    • Player maintaining close ball control under pressure
    • Defender beaten by skill move (off-balance or wrong-footed)
    • Creative ball manipulation to create space or advantage

    7. NONE
    • Normal play without significant events
    • Players positioned without clear attacking or defending action
    • Paused play, substitutions, or ordinary ball movement
    • Players walking or jogging without urgency

    Analyze all frames collectively as a sequence. If multiple events occur, select the MOST significant one (GOAL > SHOT > SAVE > THROUGH_BALL > GOOD_PASS > SKILL_MOVE).

    Return ONLY the event type without any explanation or description (e.g., "GOAL", "SHOT", "SAVE", "GOOD_PASS", "THROUGH_BALL", "SKILL_MOVE", or "NONE").
    """
    def _analyze_frames(self, frames_data, window_idx, total_windows, event_callback=None):
        """Analyze frames using OpenAI API with optimized token usage"""
        window_events = []
        
        # Skip if not enough frames
        if len(frames_data) < SEQUENCE_SIZE:
            return window_events
        
        # Get first and last timestamp for the window
        start_time = frames_data[0]['timestamp']
        end_time = frames_data[-1]['timestamp']
        start_time_str = self._format_time(start_time)
        end_time_str = self._format_time(end_time)
        
        # Create simplified message content with minimal text
        message_content = [
            {"type": "text", "text": "Classify what significant football action is happening in these frames:"}
        ]
        
        # Add fewer frames to reduce token usage
        # Select frames evenly distributed across the window
        indices = np.linspace(0, len(frames_data) - 1, min(SEQUENCE_SIZE, len(frames_data)), dtype=int)
        selected_frames = [frames_data[i] for i in indices]
        
        # Use medium detail instead of high to reduce tokens
        for frame_data in selected_frames:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_data['base64']}",
                    "detail": "high"  # Changed from high to low to reduce token usage
                }
            })
        
        # Define frame_ids for reference in results
        frame_ids = [frame_data['frame_id'] for frame_data in frames_data]
        
        # API call with max tokens limit to prevent long responses
        try:
            if DEBUG:
                print(f"Analyzing window {window_idx+1}/{total_windows}: {start_time_str} to {end_time_str} with {len(selected_frames)} frames")
            
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": self._create_player_analysis_prompt()},
                    {"role": "user", "content": message_content}
                ],
                temperature=0.2,
            )
            
            event_text = response.choices[0].message.content.strip()
            
            if DEBUG:
                print(f"API Response: {event_text}")
            
            # Skip if no event detected
            if event_text == "NONE" or "No significant actions" in event_text:
                return window_events
                
            # Create a simplified event
            event = {
                'start_time': start_time,
                'end_time': end_time,
                'event_type': event_text.strip(),
                'description': f"Event detected between {start_time_str} and {end_time_str}",
                'frame_ids': frame_ids
            }
            
            # Add event to window_events
            window_events.append(event)
            
            # Call event_callback for live updates if provided
            if event_callback and window_events:
                if DEBUG:
                    print(f"Calling event_callback with event: {event['event_type']}")
                event_callback([event])
            
            # Print event found in this window
            if window_events:
                print(f"  Window {window_idx+1}/{total_windows}: Found event {event['event_type']}")
            
        except Exception as e:
            print(f"Error analyzing window {start_time_str}-{end_time_str}: {str(e)}")
            if DEBUG:
                traceback.print_exc()
            time.sleep(2)  # Wait before trying next window
        
        return window_events

    def _parse_events(self, event_text, window_start_time, window_end_time, frame_ids):
        """Parse events from OpenAI response - simplified to handle single event type classification"""
        events = []
        
        # Skip processing if no meaningful event detected
        if not event_text or "NONE" in event_text or "No significant actions" in event_text:
            return events
            
        # Clean up the response
        event_text = event_text.strip().upper()
        
        # Map of valid event types and their normalized versions
        valid_event_map = {
            "GOAL": "GOAL",
            "GOOD_PASS": "GOOD PASS",
            "THROUGH_BALL": "THROUGH BALL",
            "SHOT": "SHOT ON TARGET",
            "SAVE": "GOOD GOALKEEPING",
            "SKILL_MOVE": "SKILL MOVE",
            "GOOD SAVE": "GOOD GOALKEEPING",
            "GOALKEEPER": "GOOD GOALKEEPING"
        }
        
        # Find matching event type
        matched_event = None
        for key, value in valid_event_map.items():
            if key in event_text:
                matched_event = value
                break
        
        # If no match found but we have some text, use it as is
        if not matched_event and len(event_text) > 0 and len(event_text) < 30:
            matched_event = event_text
        
        # Create event if we found a valid type
        if matched_event:
            events.append({
                'start_time': window_start_time,
                'end_time': window_end_time,
                'event_type': matched_event,
                'description': f"Event detected at {self._format_time(window_start_time)}",
                'frame_ids': frame_ids
            })
        
        return events

    def _merge_events(self, events):
        """Simplified merging of similar events that are close in time"""
        if not events:
            return []
        
        # Already simplified events, so minimal merging needed
        formatted_events = []
        
        # Process each event directly
        for event in events:
            start_str = self._format_time(event['start_time'])
            end_str = self._format_time(event['end_time'])
            
            # Format and add event
            formatted_events.append({
                'event_string': f"[{start_str}-{end_str}] {event['event_type']}",
                'frame_ids': event['frame_ids'],
                'start_time': event['start_time'],
                'end_time': event['end_time'],
                'event_type': event['event_type'],
                'description': f"Event detected between {start_str} and {end_str}"
            })
        
        # Sort by start time
        formatted_events.sort(key=lambda x: x['start_time'])
        
        return formatted_events

    def analyze(self, progress_callback=None, event_callback=None):
        """Main analysis pipeline with callbacks for live updates"""
        try:
            # Analyze the video
            raw_events = self.analyze_video(progress_callback, event_callback)
            
            # Format the events (this is the missing step that adds 'event_string')
            formatted_events = self._merge_events(raw_events)
            
            return formatted_events
                
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            if DEBUG:
                traceback.print_exc()
            return []
    
    def get_frame_path(self, frame_id):
        """Get path to a specific frame"""
        return os.path.join(self.temp_dir, f"{frame_id}.jpg")
    
    def cleanup(self):
        """Clean up temporary files"""
        print("\nCleaning up temporary files...")
        self.cap.release()
        
        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, file))
                except:
                    pass
            try:
                os.rmdir(self.temp_dir)
            except:
                pass
class FootballPlayerTracker:
    """
    Tracks players during detected key events using SoccerTracker
    """
    def __init__(self, video_path, yolo_model_path="yolo11m.pt", 
                 reid_model_name="osnet_x1_0", reid_weights_path="osnet_x1_0_market1501.pth",
                 device=None):
        """Initialize the player tracker"""
        self.video_path = video_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create temp directory for track outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Storage for tracked players in events
        self.event_player_tracks = {}
        self.player_thumbnails = {}  # global_id -> thumbnail image path
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap.release()
        
        # Initialize the player tracker with error handling
        try:
            # Check if model files exist first
            if not os.path.exists(yolo_model_path):
                logger.error(f"YOLO model file not found: {yolo_model_path}")
                raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")
                
            if not os.path.exists(reid_weights_path):
                logger.error(f"ReID weights file not found: {reid_weights_path}")
                raise FileNotFoundError(f"ReID weights not found: {reid_weights_path}")
                
            logger.info(f"Initializing SoccerTracker for player identification using device: {self.device}")
            self.tracker = SoccerTracker(
                yolo_model_path=yolo_model_path,
                reid_model_name=reid_model_name,
                reid_weights_path=reid_weights_path,
                device=self.device,
                ocr_interval=10  # Every 10 frames run OCR to detect jersey numbers
            )
            logger.info("SoccerTracker initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SoccerTracker: {str(e)}")
            # Create a fallback tracker or handle the error
            self.tracker = None
            raise
    
    
    
    def get_unique_players_for_event(self, event_key):
        """Get unique players identified in an event"""
        if event_key not in self.event_player_tracks:
            logger.warning(f"No tracking data found for event: {event_key}")
            return []
        
        # Get all track data for this event
        track_data = self.event_player_tracks[event_key]
        
        # Extract unique players (global IDs)
        unique_players = {}
        for frame in track_data:
            for track in frame.get('tracks', []):
                global_id = track.get('global_id')
                if global_id and global_id not in unique_players:
                    # Get the best jersey number if available
                    jersey_num = track.get('jersey_num')
                    
                    # Create player info
                    unique_players[global_id] = {
                        'global_id': global_id,
                        'jersey_num': jersey_num,
                        'thumbnail': self.player_thumbnails.get(global_id),
                        'last_position': track['bbox']
                    }
        
        logger.info(f"Found {len(unique_players)} unique players for event {event_key}")
        return list(unique_players.values())
    
    def create_player_tracking_video(self, event_key, player_id, output_path=None):
        """Create a video highlighting a specific player during an event"""
        if output_path is None:
            output_path = os.path.join(self.temp_dir, f"player_{player_id}_tracking.mp4")
        
        if event_key not in self.event_player_tracks:
            logger.error(f"No tracking data for event: {event_key}")
            return None
        
        # Get tracks for this event
        event_tracks = self.event_player_tracks[event_key]
        
        if not event_tracks:
            logger.error(f"Empty tracking data for event: {event_key}")
            return None
            
        # Debug information
        logger.info(f"Creating video for player {player_id} in event {event_key}")
        logger.info(f"Number of frame records: {len(event_tracks)}")
        
        # First check frame paths
        valid_frame_paths = []
        for frame_data in event_tracks:
            if 'frame_path' in frame_data and os.path.exists(frame_data['frame_path']):
                valid_frame_paths.append(frame_data['frame_path'])
        
        logger.info(f"Valid frame paths: {len(valid_frame_paths)}/{len(event_tracks)}")
        
        if not valid_frame_paths:
            logger.error("No valid frame paths found")
            return None
        
        # Check if the player appears in any frames
        frames_with_player = 0
        for frame_data in event_tracks:
            for track in frame_data.get('tracks', []):
                if track.get('global_id') == player_id:
                    frames_with_player += 1
                    break
        
        logger.info(f"Frames with player {player_id}: {frames_with_player}/{len(event_tracks)}")
        
        if frames_with_player == 0:
            logger.error(f"Player {player_id} not found in any frames")
            return None
        
        # Create video writer with first valid frame dimensions
        try:
            first_frame = cv2.imread(valid_frame_paths[0])
            if first_frame is None:
                logger.error(f"Failed to read first frame: {valid_frame_paths[0]}")
                return None
                
            h, w = first_frame.shape[:2]
            
            # Check if output directory exists
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Create video writer - try different codecs if needed
            try:
                # First try mp4v codec
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, self.fps / TRACK_INTERVAL, (w, h))
                
                # Check if writer is initialized
                if not writer.isOpened():
                    # Try XVID as fallback
                    logger.warning("mp4v codec failed, trying XVID")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    writer = cv2.VideoWriter(output_path, fourcc, self.fps / TRACK_INTERVAL, (w, h))
                    
                    if not writer.isOpened():
                        logger.error("Failed to initialize video writer with both mp4v and XVID codecs")
                        return None
            except Exception as e:
                logger.error(f"Error creating video writer: {str(e)}")
                return None
            
            # Track history for drawing trails
            track_history = []
            frames_written = 0
            
            # Process each frame
            logger.info(f"Writing video to: {output_path}")
            
            for frame_data in event_tracks:
                # Get frame path
                frame_path = frame_data.get('frame_path')
                if not frame_path or not os.path.exists(frame_path):
                    continue
                    
                # Read the annotated frame
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                
                # Find this player in current frame tracks
                player_track = None
                for track in frame_data.get('tracks', []):
                    if track.get('global_id') == player_id:
                        player_track = track
                        break
                
                try:
                    # Create highlighted frame
                    if player_track:
                        # Add to track history (for trail)
                        center_x = (player_track['bbox'][0] + player_track['bbox'][2]) // 2
                        center_y = (player_track['bbox'][1] + player_track['bbox'][3]) // 2
                        track_history.append((center_x, center_y))
                        
                        # Limit history length
                        if len(track_history) > 30:
                            track_history = track_history[-30:]
                        
                        # Highlight player
                        highlighted_frame = self._highlight_player(frame, player_track, track_history)
                        writer.write(highlighted_frame)
                    else:
                        # Write the original frame if player not found - add "Player not in frame" text
                        text_frame = frame.copy()
                        cv2.putText(
                            text_frame,
                            "Player not in frame",
                            (w//2 - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2
                        )
                        writer.write(text_frame)
                    
                    frames_written += 1
                    
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                    # Continue to next frame
            
            # Release the writer
            writer.release()
            
            logger.info(f"Video creation complete. Wrote {frames_written} frames.")
            
            # Check if the video was created and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Successfully created video: {output_path}, size: {os.path.getsize(output_path)} bytes")
                return output_path
            else:
                logger.error(f"Video file missing or empty: {output_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating tracking video: {str(e)}")
            traceback.print_exc()
            return None
    
    def track_players_for_event(self, event, progress_callback=None):
        """Track players directly from original video for an event with extended buffer"""
        # Ensure we have the needed event data
        if 'event_string' not in event:
            if 'start_time' in event and 'end_time' in event and 'event_type' in event:
                # Create event_string on the fly
                start_str = f"{int(event['start_time'] // 60):02d}:{int(event['start_time'] % 60):02d}"
                end_str = f"{int(event['end_time'] // 60):02d}:{int(event['end_time'] % 60):02d}"
                event['event_string'] = f"[{start_str}-{end_str}] {event['event_type']}"
            else:
                logger.error("Event data is missing required fields")
                return []
                
        # Skip if we've already tracked this event
        event_key = event['event_string']
        if event_key in self.event_player_tracks:
            logger.info(f"Using cached tracking data for event: {event_key}")
            return self.event_player_tracks[event_key]
        
        # Extract time range for this event
        start_time = event['start_time']
        end_time = event['end_time']
        
        # Add an extended buffer around the event (10 seconds before and after)
        start_time = max(0, start_time - 10)  # Changed from 2 to 10 seconds
        end_time = end_time + 10  # Changed from 2 to 10 seconds
        
        # Convert to frame numbers
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        # Verify that the video file exists
        if not os.path.exists(self.video_path):
            logger.error(f"Video file not found: {self.video_path}")
            return []
            
        # Track players in these frames
        tracks = []
        player_thumbnails = {}
        
        # Open original video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file for tracking: {self.video_path}")
            return []
        
        # Calculate total frames to process
        total_frames = end_frame - start_frame
        processed_frames = 0
        
        # Process frames from original video
        try:
            logger.info(f"Starting player tracking for event: {event_key}")
            logger.info(f"Processing frames {start_frame} to {end_frame} with interval {TRACK_INTERVAL}")
            
            for frame_idx in range(start_frame, end_frame + 1, TRACK_INTERVAL):
                # Update progress if callback provided
                if progress_callback and total_frames > 0:
                    progress = processed_frames / ((total_frames / TRACK_INTERVAL) or 1)
                    progress_callback(min(progress, 0.99))  # Cap at 99% until fully complete
                
                # Read frame directly from original video
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {frame_idx}")
                    break
                
                # Process frame using player tracker
                try:
                    annotated_frame, tracks_for_frame = self.tracker.process_frame(frame)
                    
                    # Save annotated frame
                    safe_event_key = event_key.replace(':', '-').replace(' ', '_').replace('[', '').replace(']', '')
                    frame_filename = f"track_{safe_event_key}_{frame_idx}.jpg"
                    frame_path = os.path.join(self.temp_dir, frame_filename)
                    cv2.imwrite(frame_path, annotated_frame)
                    
                    # Log tracks found in this frame
                    if tracks_for_frame:
                        logger.debug(f"Frame {frame_idx}: Found {len(tracks_for_frame)} tracks")
                    
                    # Store player thumbnails
                    for track in tracks_for_frame:
                        global_id = track.get('global_id')
                        if global_id and global_id not in player_thumbnails:
                            try:
                                # Extract player thumbnail
                                bbox = track['bbox']
                                x1, y1, x2, y2 = bbox
                                
                                # Validate bbox coordinates
                                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                                    logger.warning(f"Invalid bbox for player {global_id}: {bbox}")
                                    continue
                                
                                # Add padding
                                padding = 10
                                x1 = max(0, x1 - padding)
                                y1 = max(0, y1 - padding)
                                x2 = min(frame.shape[1], x2 + padding)
                                y2 = min(frame.shape[0], y2 + padding)
                                
                                # Get thumbnail from original frame
                                player_thumbnail = frame[y1:y2, x1:x2]
                                if player_thumbnail.size > 0:
                                    thumbnail_filename = f"player_{global_id}_{uuid.uuid4()}.jpg"
                                    thumbnail_path = os.path.join(self.temp_dir, thumbnail_filename)
                                    cv2.imwrite(thumbnail_path, player_thumbnail)
                                    player_thumbnails[global_id] = thumbnail_path
                                    logger.debug(f"Created thumbnail for player {global_id}")
                            except Exception as e:
                                logger.error(f"Error creating thumbnail for player {global_id}: {str(e)}")
                    
                    # Store tracks for this frame
                    tracks.append({
                        'frame_idx': frame_idx,
                        'frame_path': frame_path,
                        'tracks': tracks_for_frame
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                    if DEBUG:
                        traceback.print_exc()
                
                processed_frames += 1
        
        except Exception as e:
            logger.error(f"Error tracking players for event {event_key}: {str(e)}")
            if DEBUG:
                traceback.print_exc()
        
        finally:
            cap.release()
            
        # Verify we have some tracking data
        if not tracks:
            logger.warning(f"No tracking data generated for event {event_key}")
            
        # Store tracks for this event
        self.event_player_tracks[event_key] = tracks
        self.player_thumbnails.update(player_thumbnails)
        
        # Log summary
        logger.info(f"Tracking complete for event {event_key}: {len(tracks)} frames with {len(player_thumbnails)} unique players")
        
        # Final progress update
        if progress_callback:
            progress_callback(1.0)
        
        return tracks

    def _highlight_player(self, frame, player_track, track_history):
        """Highlight specific player with circle and lighter blur on other areas"""
        try:
            # Create copy of frame
            result = frame.copy()
            
            # Create mask for the player to highlight
            mask = np.zeros_like(frame, dtype=np.uint8)
            
            # Get player bounding box
            x1, y1, x2, y2 = player_track['bbox']
            
            # Get player center and radius
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = max((x2 - x1) // 2, (y2 - y1) // 2) + 20  # Add padding
            
            # Draw filled circle for the player on the mask
            cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
            
            # Apply lighter blur to the background - reducing blur kernel size
            blurred = cv2.GaussianBlur(result, (15, 15), 0)  # Reduced from 25 to 15
            
            # Combine the original (highlighted player) with blurred background
            mask_inv = cv2.bitwise_not(mask)
            fg = cv2.bitwise_and(result, mask)
            bg = cv2.bitwise_and(blurred, mask_inv)
            result = cv2.add(fg, bg)
            
            # Draw a circle around the player
            cv2.circle(result, (center_x, center_y), radius, (0, 165, 255), 3)
            
            # Add player information
            jersey_num = player_track.get('jersey_num')
            gid = player_track.get('global_id', 'Unknown')
            
            label = f"Player {gid}"
            if jersey_num:
                label = f"Player #{jersey_num}"
                
            # Add text with shadow effect
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            text_thickness = 2
            shadow_offset = 2
            
            # Shadow
            cv2.putText(result, label, (center_x - 70 + shadow_offset, y1 - 20 + shadow_offset), 
                        font, font_scale, (0, 0, 0), text_thickness + 1)
            # Text
            cv2.putText(result, label, (center_x - 70, y1 - 20), 
                        font, font_scale, (0, 165, 255), text_thickness)
            
            return result
            
        except Exception as e:
            logger.error(f"Error highlighting player: {str(e)}")
            # Return original frame if highlighting fails
            return frame
    
    def cleanup(self):
        """Clean up temporary files"""
        logger.info("\nCleaning up temporary tracking files...")
        
        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, file))
                except Exception as e:
                    logger.error(f"Error removing temp file: {str(e)}")
            try:
                os.rmdir(self.temp_dir)
            except Exception as e:
                logger.error(f"Error removing temp directory: {str(e)}")

    def generate_player_highlights(self, output_dir="highlights"):
        """Generate highlight data and videos for tracked players"""
        if not self.tracker:
            logger.error("Tracker not initialized")
            return None
            
        try:
            # Make sure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate highlight data
            highlight_data_path = os.path.join(output_dir, "highlight_data.json")
            highlight_data = self.tracker.generate_player_highlight_data(
                output_path=highlight_data_path,
                min_segment_duration=1.0,
                min_total_duration=3.0
            )
            
            logger.info(f"Generated highlight data for {len(highlight_data)} players")
            return highlight_data
            
        except Exception as e:
            logger.error(f"Error generating player highlights: {str(e)}")
            if DEBUG:
                traceback.print_exc()
            return None
            
    def generate_player_trajectory(self, player_id, output_path=None):
        """Generate trajectory visualization for a specific player"""
        if not self.tracker:
            logger.error("Tracker not initialized")
            return None
            
        try:
            # Create output path if not provided
            if not output_path:
                if not os.path.exists("trajectories"):
                    os.makedirs("trajectories", exist_ok=True)
                output_path = f"trajectories/player_{player_id}_trajectory.jpg"
                
            # Generate trajectory for just this player
            trajectory_img = self.tracker.generate_trajectory(
                output_path=output_path,
                frame_size=(self.frame_width, self.frame_height),
                selected_players=[player_id]  # Only include this player
            )
            
            if trajectory_img is not None:
                logger.info(f"Generated trajectory for player {player_id}")
                return output_path
            else:
                logger.error(f"Failed to generate trajectory for player {player_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating player trajectory: {str(e)}")
            if DEBUG:
                traceback.print_exc()
            return None

    def generate_player_heatmap(self, player_id, output_path=None):
        """Generate heatmap visualization for a specific player that matches trajectory style"""
        if not self.tracker:
            logger.error("Tracker not initialized")
            return None
        
        try:
            # Create output path if not provided
            if not output_path:
                if not os.path.exists("heatmaps"):
                    os.makedirs("heatmaps", exist_ok=True)
                output_path = f"heatmaps/player_{player_id}_heatmap.jpg"
            
            # Check if player exists in track_history_viz
            if not hasattr(self.tracker, 'track_history_viz') or player_id not in self.tracker.track_history_viz:
                logger.warning(f"Player {player_id} not found in trajectory history")
                
                # Fall back to event tracking data if trajectory history is not available
                player_positions = []
                for event_key, event_tracks in self.event_player_tracks.items():
                    for frame_data in event_tracks:
                        for track in frame_data.get('tracks', []):
                            if track.get('global_id') == player_id:
                                bbox = track.get('bbox')
                                if bbox and len(bbox) == 4:
                                    center_x = (bbox[0] + bbox[2]) / 2
                                    center_y = (bbox[1] + bbox[3]) / 2
                                    player_positions.append((center_x, center_y))
                                    break
                
                if not player_positions:
                    logger.error(f"No position data available for player {player_id}")
                    return None
            else:
                # Get trajectory points from the same source used for trajectory visualization
                trajectory_points = []
                for entry in self.tracker.track_history_viz[player_id]:
                    if entry.get('x') is not None and entry.get('y') is not None:
                        trajectory_points.append((entry['x'], entry['y']))
                
                if not trajectory_points:
                    logger.error(f"No trajectory points found for player {player_id}")
                    return None
                
                player_positions = trajectory_points
            
            logger.info(f"Generating heatmap using {len(player_positions)} position points")
            
            # Create visualization canvas with the same dimensions as trajectory visualization
            w_viz, h_viz = 1280, 720
            
            # Scale coordinates to match visualization canvas
            scale_x, scale_y = w_viz / self.frame_width, h_viz / self.frame_height
            
            # Create grid for heatmap that matches visualization dimensions
            grid_size = 20  # Size of each grid cell
            height_cells = h_viz // grid_size
            width_cells = w_viz // grid_size
            
            # Initialize grid with zeros
            heatmap_grid = np.zeros((height_cells, width_cells))
            
            # Count occurrences in each grid cell (with scaled coordinates)
            for x, y in player_positions:
                # Scale coordinates
                scaled_x = x * scale_x
                scaled_y = y * scale_y
                
                if 0 <= scaled_x < w_viz and 0 <= scaled_y < h_viz:
                    grid_x, grid_y = int(scaled_x) // grid_size, int(scaled_y) // grid_size
                    if 0 <= grid_x < width_cells and 0 <= grid_y < height_cells:
                        heatmap_grid[grid_y, grid_x] += 1
            
            # Create the heatmap visualization
            plt.figure(figsize=(12, 8))
            
            # Create field background (matching trajectory visualization)
            field_img = np.ones((h_viz, w_viz, 3), dtype=np.uint8) * 255  # White background
            
            # Draw field markings (same as in trajectory visualization)
            # Draw Field
            cv2.rectangle(field_img, (50, 50), (w_viz-50, h_viz-50), (0, 100, 0), 2)
            cv2.line(field_img, (w_viz//2, 50), (w_viz//2, h_viz-50), (0, 100, 0), 2)
            cv2.circle(field_img, (w_viz//2, h_viz//2), int(9.15 * scale_y * 10), (0, 100, 0), 2)  # Center circle
            
            # Convert to RGB for matplotlib
            field_img_rgb = cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB)
            
            # Show the field
            plt.imshow(field_img_rgb)
            
            # Apply Gaussian smoothing to make the heatmap look nicer
            # Use different sigma values for smoothness
            heatmap_grid_smooth = gaussian_filter(heatmap_grid, sigma=2.0)
            
            # Plot the heatmap with appropriate colormap
            # Use jet colormap for better visibility on the field
            heatmap = plt.imshow(heatmap_grid_smooth, cmap='hot', alpha=0.7, 
                                interpolation='bilinear', extent=[0, w_viz, h_viz, 0])
            
            # Add colorbar with better formatting
            cbar = plt.colorbar(heatmap, label='Position Frequency', shrink=0.8)
            cbar.ax.tick_params(labelsize=10)
            
            # Get player info for title
            player_label = f"Player {player_id}"
            if hasattr(self.tracker, 'reid'):
                jersey_num, _ = self.tracker.reid.get_best_jersey(player_id)
                if jersey_num:
                    player_label = f"Player #{jersey_num}"
            
            # Add title and timestamp
            plt.title(f"{player_label} Movement Heatmap", fontsize=14, fontweight='bold')
            plt.text(10, h_viz-10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                    fontsize=8, color='black')
            
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated heatmap for player {player_id}: {output_path}")
            return output_path
                
        except Exception as e:
            logger.error(f"Error generating player heatmap: {str(e)}")
            if DEBUG:
                traceback.print_exc()
            return None

# Main function to use the analyzer and tracker
def analyze_football_video(video_path, output_path=None, progress_callback=None, event_callback=None):
    """Analyze football video and save results with callbacks for live updates"""
    if not output_path:
        output_path = os.path.splitext(video_path)[0] + "_events.txt"
    
    analyzer = FootballPlayerAnalyzer(video_path)
    events = analyzer.analyze(progress_callback, event_callback)
    
    # Save results with error handling for missing keys
    with open(output_path, 'w') as f:
        f.write("FOOTBALL PLAYER PERFORMANCE EVENTS\n")
        f.write("=================================\n\n")
        for event in events:
            # Handle missing 'event_string' key
            if 'event_string' not in event:
                start_str = analyzer._format_time(event['start_time'])
                end_str = analyzer._format_time(event['end_time'])
                event_string = f"[{start_str}-{end_str}] {event['event_type']}"
                f.write(f"{event_string}\n")
            else:
                f.write(f"{event['event_string']}\n")
            
            f.write(f"  Frames: {', '.join(event['frame_ids'][:5])}...\n\n")
    
    # Save events to JSON for later loading - use consistent naming
    json_output_path = "events_data.json"
    try:
        with open(json_output_path, 'w') as f:
            json.dump(events, f)
        print(f"Events saved to {json_output_path} for future loading")
    except Exception as e:
        print(f"Error saving events to JSON: {str(e)}")
    
    print(f"\nResults saved to {output_path}")
    print("\nDetected Events:")
    for event in events:
        # Handle missing 'event_string' key when printing
        if 'event_string' not in event:
            start_str = analyzer._format_time(event['start_time'])
            end_str = analyzer._format_time(event['end_time'])
            event_string = f"[{start_str}-{end_str}] {event['event_type']}"
            print(event_string)
        else:
            print(event['event_string'])
        
        print(f"  Frames: {', '.join(event['frame_ids'][:5])}...")
        print()
    
    return events, analyzer


# Enhanced Streamlit app function
def streamlit_app():
    st.set_page_config(layout="wide", page_title="Football Analytics Platform", page_icon="⚽")
    
    # Style customizations
    st.markdown("""
    <style>
    .event-box {
        border-left: 4px solid #0078ff;
        padding: 10px;
        margin: 10px 0;
        background-color: #f8f9fa;
        border-radius: 4px;
    }
    .player-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .player-card:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .selected-player {
        border: 3px solid #ff6b00;
        box-shadow: 0 0 10px rgba(255,107,0,0.5);
    }
    .player-icon {
        width: 100%;
        max-width: 80px;
        margin: 0 auto;
        display: block;
        border-radius: 50%;
        border: 2px solid #ddd;
    }
    .player-highlights-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        transition: transform 0.2s;
        background-color: white;
    }
    .player-highlights-card:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .player-name-input {
        margin-top: 5px;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("⚽ Football Analytics Platform")
    st.write("Analyze key events, track players, and visualize performance")
    
    # Sidebar for app modes
    st.sidebar.title("Analytics Options")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Event Detection", "Player Tracking", "Team Analysis"])
    
    # Create tabs for the main app sections
    if app_mode == "Event Detection":
        tabs = st.tabs(["Upload & Analysis", "Event Viewer"])
        upload_tab, event_tab = tabs
    elif app_mode == "Player Tracking":
        tabs = st.tabs(["Player Tracking", "Player Highlights", "Player Stats"])
        tracking_tab, highlights_tab, stats_tab = tabs
    else:  # Team Analysis
        tabs = st.tabs(["Team Overview", "Tactical Analysis"])
        team_tab, tactical_tab = tabs
    
    # Global variables for event tracking
    if 'events' not in st.session_state:
        st.session_state.events = []
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
    if 'player_tracker' not in st.session_state:
        st.session_state.player_tracker = None
    
    if 'current_time' not in st.session_state:
        st.session_state.current_time = 0
    
    if 'selected_player' not in st.session_state:
        st.session_state.selected_player = None
    
    if 'selected_event' not in st.session_state:
        st.session_state.selected_event = None
    
    if 'tracking_progress' not in st.session_state:
        st.session_state.tracking_progress = 0
        
    if 'tracking_error' not in st.session_state:
        st.session_state.tracking_error = None
        
    if 'player_data' not in st.session_state:
        st.session_state.player_data = {}
        
    if 'player_highlights' not in st.session_state:
        st.session_state.player_highlights = {}
        
    if 'player_trajectories' not in st.session_state:
        st.session_state.player_trajectories = {}
    
    # File uploader for video
    if app_mode == "Event Detection":
        with upload_tab:
            st.subheader("Upload Match Video")
            uploaded_file = st.file_uploader("Upload a football match video", type=["mp4", "avi", "mov"])
            
            if uploaded_file is not None:
                # Save the uploaded file temporarily
                temp_video_path = "temp_video.mp4"
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Display the video using Streamlit's native video player
                st.video(temp_video_path)
                
                # Check for existing analysis in our database
                existing_analysis = check_existing_analysis(temp_video_path)
                
                # Create columns for buttons
                col1, col2, col3 = st.columns([1, 1, 1])
                
                # Tracking pipeline status storage
                if 'tracking_process' not in st.session_state:
                    st.session_state.tracking_process = None
                
                if 'tracking_output_dir' not in st.session_state:
                    st.session_state.tracking_output_dir = None
                
                with col1:
                    analyze_button = st.button("Start Event Analysis", type="primary")
                
                with col2:
                    # Add button to run tracking pipeline
                    tracking_button = st.button("Run Full Tracking Pipeline", type="secondary")
                
                with col3:
                    # Check if events file exists in different possible locations
                    events_file_paths = ["events_data.json"]
                    
                    # Add path from existing analysis if available
                    if existing_analysis and existing_analysis.get('events_json_path'):
                        events_json_path = existing_analysis['events_json_path']
                        # Check if it's a string and not NaN
                        if isinstance(events_json_path, str) and events_json_path and os.path.exists(events_json_path):
                            events_file_paths.append(events_json_path)
                    
                    # Add path from output directory if tracking has run
                    if st.session_state.tracking_output_dir and os.path.exists(os.path.join(st.session_state.tracking_output_dir, "highlight_data.json")):
                        events_file_paths.append(os.path.join(st.session_state.tracking_output_dir, "highlight_data.json"))
                    
                    events_file_exists = any(os.path.exists(path) for path in events_file_paths)
                    
                    if events_file_exists:
                        load_events_button = st.button("Load Existing Analysis")
                        if st.session_state.events:
                            st.success(f"✅ {len(st.session_state.events)} events loaded/detected")
                    else:
                        load_events_button = False
                        if st.session_state.events:
                            st.success(f"✅ {len(st.session_state.events)} events detected")
                
                # Information about existing analysis
                if existing_analysis:
                    st.info(f"📊 Previous analysis found for this video from {existing_analysis['analysis_timestamp']}. You can load existing data or reanalyze.")
                    
                    with st.expander("View Previous Analysis Details"):
                        st.write(f"Output Directory: {existing_analysis['output_directory']}")
                        st.write(f"Tracking Completed: {'Yes' if existing_analysis['tracking_complete'] else 'No'}")
                        st.write(f"Highlights Generated: {'Yes' if existing_analysis['highlights_generated'] else 'No'}")
                        
                        # Check if the directory exists and what files it contains
                        if os.path.exists(existing_analysis['output_directory']):
                            files = os.listdir(existing_analysis['output_directory'])
                            st.write(f"Files in output directory: {len(files)}")
                            
                            # Display a few key files
                            key_files = [f for f in files if f.endswith('.json') or f.endswith('.csv') or f.endswith('.mp4')]
                            if key_files:
                                st.write("Key files:")
                                for file in key_files[:5]:  # Show first 5 key files
                                    st.code(file)
                                
                                if len(key_files) > 5:
                                    st.write(f"...and {len(key_files) - 5} more")
                        else:
                            st.warning("Output directory not found. Some data may be missing.")
                
                # Progress bar placeholder
                progress_bar = st.empty()
                status_text = st.empty()
                
                # Check if tracking is in progress
                if st.session_state.tracking_process is not None:
                    with st.spinner("Tracking in progress..."):
                        # Check tracking status
                        running, status, output = check_tracking_status(st.session_state.tracking_process)
                        
                        if running:
                            # Still running
                            status_text.info(f"Tracking Status: {status}")
                            
                            # Show the most recent output to give feedback
                            if output:
                                st.text_area("Latest output", output, height=150)
                                
                            # Button to manually refresh status
                            if st.button("Refresh Status"):
                                st.rerun()
                        else:
                            # Tracking finished
                            if status == "Completed":
                                status_text.success("✅ Tracking pipeline completed successfully!")
                                
                                # Save the metadata to our database
                                save_analysis_metadata(
                                    temp_video_path, 
                                    st.session_state.tracking_output_dir,
                                    os.path.join(st.session_state.tracking_output_dir, "highlight_data.json"),
                                    tracking_complete=True,
                                    highlights_generated=True
                                )
                                
                                # Clear the process
                                st.session_state.tracking_process = None
                                
                                # Add button to view results
                                if st.button("View Analysis Results"):
                                    # Set selected tab to player highlights
                                    st.session_state.app_mode = "Player Tracking"
                                    st.rerun()
                            else:
                                # Failed
                                status_text.error(f"❌ Tracking failed: {output}")
                                st.session_state.tracking_process = None
                
                # Start tracking if button clicked
                elif tracking_button:
                        status_text.info("Starting full tracking pipeline...")
                        
                        # Create output directory based on video hash
                        video_hash = get_video_hash(temp_video_path)
                        output_dir = f"output_{video_hash}"
                        
                        # Use our direct function instead of subprocess
                        with st.spinner("Running tracking pipeline... This may take several minutes."):
                            success, message, output_dir = run_tracking_pipeline(
                                temp_video_path,
                                output_dir=output_dir,
                                device="cuda" if torch.cuda.is_available() else "cpu"
                            )
                            
                            if success:
                                # Store the output directory for future reference
                                st.session_state.tracking_output_dir = output_dir
                                
                                # Save metadata to database
                                save_analysis_metadata(
                                    temp_video_path, 
                                    output_dir,
                                    events_json_path=os.path.join(output_dir, "highlight_data.json"),
                                    tracking_complete=True,
                                    highlights_generated=True
                                )
                                
                                # Load the events from the output directory
                                events_path = os.path.join(output_dir, "highlight_data.json")
                                if os.path.exists(events_path):
                                    loaded_events = load_events_from_file(events_path)
                                    st.session_state.events = loaded_events
                                    
                                    # Initialize the player tracker with the processed data
                                    try:
                                        player_tracker = FootballPlayerTracker(
                                            temp_video_path,
                                            yolo_model_path="yolov8n.pt",
                                            reid_weights_path="osnet_x1_0_market1501.pth"
                                        )
                                        player_tracker.output_dir = output_dir
                                        st.session_state.player_tracker = player_tracker
                                    except Exception as e:
                                        status_text.error(f"Error initializing player tracker: {str(e)}")
                                
                                status_text.success(f"✅ Tracking pipeline completed successfully! Results saved to: {output_dir}")
                                
                                # Add button to view results
                                if st.button("View Analysis Results"):
                                    # Set selected tab to player highlights
                                    st.session_state.app_mode = "Player Tracking"
                                    st.rerun()
                            else:
                                status_text.error(f"❌ Tracking failed: {message}")
                        
                # Start analysis if button clicked
                elif analyze_button:
                    progress_value = progress_bar.progress(0)
                    status_text.text("Starting analysis...")
                    
                    # Clear previous events
                    st.session_state.events = []
                    
                    # Function to update the progress bar
                    def update_progress(progress):
                        progress_value.progress(progress)
                    
                    # Function to update events in real-time
                    def update_events(new_events):
                        """Function to update events in real-time with proper formatting"""
                        # Convert new events to the formatted version
                        formatted_events = []
                        for event in new_events:
                            if 'event_string' not in event:
                                # Create event_string on the fly
                                start_str = f"{int(event['start_time'] // 60):02d}:{int(event['start_time'] % 60):02d}"
                                end_str = f"{int(event['end_time'] // 60):02d}:{int(event['end_time'] % 60):02d}"
                                formatted_events.append({
                                    'event_string': f"[{start_str}-{end_str}] {event['event_type']}",
                                    'frame_ids': event['frame_ids'],
                                    'start_time': event['start_time'],
                                    'end_time': event['end_time'],
                                    'event_type': event['event_type'],
                                    'description': event['description']
                                })
                            else:
                                formatted_events.append(event)
                        
                        # Update the session state with the new events
                        st.session_state.events.extend(formatted_events)
                    
                    try:
                        # Run the analysis
                        status_text.text("Analyzing video for key events...")
                        
                        # Determine output path - use tracking output directory if available
                        events_output_path = "player_performance.txt"
                        events_json_path = "events_data.json"
                        
                        if st.session_state.tracking_output_dir:
                            events_output_path = os.path.join(st.session_state.tracking_output_dir, "player_performance.txt")
                            events_json_path = os.path.join(st.session_state.tracking_output_dir, "events_data.json")
                        elif existing_analysis and os.path.exists(existing_analysis['output_directory']):
                            events_output_path = os.path.join(existing_analysis['output_directory'], "player_performance.txt")
                            events_json_path = os.path.join(existing_analysis['output_directory'], "events_data.json")
                        
                        events, analyzer = analyze_football_video(
                            temp_video_path,
                            events_output_path,
                            progress_callback=update_progress,
                            event_callback=update_events
                        )
                        
                        # Store analyzer for frame access
                        st.session_state.analyzer = analyzer
                        
                        # Update session state with final events
                        if events and not st.session_state.events:
                            st.session_state.events = events
                        
                        # Initialize player tracker
                        status_text.text("Initializing player tracking...")
                        try:
                            # Determine model paths - use from tracking output if available
                            yolo_path = "yolov8n.pt"
                            reid_path = "osnet_x1_0_market1501.pth"
                            
                            if st.session_state.tracking_output_dir:
                                output_yolo = os.path.join(st.session_state.tracking_output_dir, "yolov8n.pt")
                                output_reid = os.path.join(st.session_state.tracking_output_dir, "osnet_x1_0_market1501.pth")
                                
                                if os.path.exists(output_yolo):
                                    yolo_path = output_yolo
                                if os.path.exists(output_reid):
                                    reid_path = output_reid
                            
                            # Check if model files exist
                            if not os.path.exists(yolo_path):
                                status_text.warning(f"YOLO model file not found: {yolo_path}")
                                yolo_path = st.text_input("Enter path to YOLO model file:", "yolov8n.pt")
                                
                            if not os.path.exists(reid_path):
                                status_text.warning(f"ReID model file not found: {reid_path}")
                                reid_path = st.text_input("Enter path to ReID model file:", "osnet_x1_0_market1501.pth")
                            
                            player_tracker = FootballPlayerTracker(
                                temp_video_path,
                                yolo_model_path=yolo_path,
                                reid_weights_path=reid_path
                            )
                            
                            # Set output directory for highlights search
                            if st.session_state.tracking_output_dir:
                                player_tracker.output_dir = st.session_state.tracking_output_dir
                            elif existing_analysis and os.path.exists(existing_analysis['output_directory']):
                                player_tracker.output_dir = existing_analysis['output_directory']
                            
                            st.session_state.player_tracker = player_tracker
                            status_text.success("Player tracking initialized!")
                            
                            # Save analysis metadata
                            if st.session_state.tracking_output_dir:
                                output_dir = st.session_state.tracking_output_dir
                            elif existing_analysis:
                                output_dir = existing_analysis['output_directory']
                            else:
                                # Create directory based on video hash
                                video_hash = get_video_hash(temp_video_path)
                                output_dir = f"output_{video_hash}"
                                os.makedirs(output_dir, exist_ok=True)
                            
                            save_analysis_metadata(
                                temp_video_path,
                                output_dir,
                                events_json_path=events_json_path,
                                tracking_complete=False  # We didn't run full tracking yet
                            )
                            
                        except Exception as e:
                            status_text.error(f"Error initializing player tracking: {str(e)}")
                            st.session_state.tracking_error = str(e)
                            if DEBUG:
                                st.error(traceback.format_exc())
                        
                        progress_bar.progress(100)
                        status_text.success("Analysis complete! View results in the Event Viewer tab.")
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        if DEBUG:
                            st.error(traceback.format_exc())
                
        # Event viewer tab
        with event_tab:
            if st.session_state.events and st.session_state.analyzer:
                st.subheader("Detected Events")
                
                # Filter by event type
                event_types = ["All Types"] + list(set(e['event_type'] for e in st.session_state.events))
                selected_type = st.selectbox("Filter by event type:", event_types)
                
                # Display events
                filtered_events = st.session_state.events
                if selected_type != "All Types":
                    filtered_events = [e for e in st.session_state.events if e['event_type'] == selected_type]
                
                st.write(f"Showing {len(filtered_events)} events")
                
                for idx, event in enumerate(filtered_events):
                    event_key = event['event_string']
                    
                    # Create a unique key for this event
                    event_id = f"event_{idx}"
                    
                    # Custom styled event box
                    st.markdown(f"""
                    <div class="event-box">
                        <h4>{event['event_string']}</h4>
                        <p>{event.get('description', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        # Show a few frames from this event
                        frames_to_show = event['frame_ids'][:3]
                        if frames_to_show:
                            frame_cols = st.columns(len(frames_to_show))
                            for i, frame_id in enumerate(frames_to_show):
                                frame_path = st.session_state.analyzer.get_frame_path(frame_id)
                                if os.path.exists(frame_path):
                                    frame_cols[i].image(frame_path, caption=f"Frame {i+1}", use_container_width=True)
                    
                    with col2:
                        # Add button to seek to this event in the video
                        if st.button(f"📹 View in Video", key=f"view_{idx}"):
                            st.session_state.current_time = event['start_time']
                            # Force video reload at the correct position
                            st.rerun()
                    
                    with col3:
                        # Add button to track players for this event
                        if st.button(f"👤 Track Players", key=f"track_{idx}"):
                            st.session_state.selected_event = event_key
                            # Switch to Player Tracking tab
                            st.rerun()
                    
                    st.write("---")
            else:
                st.info("No events detected yet. Go to the Upload & Analysis tab to analyze a video.")
    
    # Player Tracking Tab
    elif app_mode == "Player Tracking":
        with tracking_tab:
            if not st.session_state.events:
                st.info("Please analyze a video first in the Event Detection mode.")
            elif st.session_state.tracking_error:
                st.error(f"Player tracking is not available due to an error: {st.session_state.tracking_error}")
                st.info("Common issues include missing model files. Make sure you have both:")
                st.code("yolov8n.pt\nosnet_x1_0_market1501.pth")
                
                if st.button("Reset Tracking Error and Try Again"):
                    st.session_state.tracking_error = None
                    st.rerun()
            elif not st.session_state.player_tracker:
                st.info("Player tracking not initialized. Please analyze a video first.")
                
                # Offer to initialize tracker manually
                if st.button("Initialize Player Tracker"):
                    try:
                        # Check for video path
                        temp_video_path = "temp_video.mp4"
                        if not os.path.exists(temp_video_path):
                            st.error(f"Video file not found: {temp_video_path}")
                        else:
                            player_tracker = FootballPlayerTracker(temp_video_path)
                            st.session_state.player_tracker = player_tracker
                            st.success("Player tracking initialized successfully!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error initializing player tracking: {str(e)}")
                        st.code(traceback.format_exc())
            else:
                st.subheader("Player Tracking for Key Events")
                
                # Select an event to track players
                event_options = [event['event_string'] for event in st.session_state.events]
                
                # Set default selection to the one from event viewer if available
                default_index = 0
                if st.session_state.selected_event in event_options:
                    default_index = event_options.index(st.session_state.selected_event)
                
                selected_event = st.selectbox(
                    "Select an event to track players:",
                    event_options, 
                    index=default_index
                )
                
                # Store the selected event
                st.session_state.selected_event = selected_event
                
                # Get the event data
                event_data = None
                for event in st.session_state.events:
                    if event['event_string'] == selected_event:
                        event_data = event
                        break
                
                if not event_data:
                    st.error("Event data not found.")
                else:
                    # Check if we've already tracked this event
                    if selected_event not in st.session_state.player_tracker.event_player_tracks:
                        st.info("⏳ Tracking players for this event from original video (this may take a few moments)...")
                        
                        # Progress bar for tracking
                        tracking_progress = st.progress(0, "Initializing player tracking...")
                        
                        def update_tracking_progress(progress):
                            tracking_progress.progress(progress, f"Tracking players: {int(progress * 100)}% complete")
                        
                        try:
                            # Track players for this event
                            tracks = st.session_state.player_tracker.track_players_for_event(
                                event_data, 
                                update_tracking_progress
                            )
                            
                            if tracks:
                                tracking_progress.progress(1.0, "✅ Tracking complete!")
                                st.success(f"Successfully tracked players across {len(tracks)} frames")
                            else:
                                tracking_progress.progress(1.0, "⚠️ No tracking data generated")
                                st.warning("No player tracking data was generated. This may be due to:")
                                st.markdown("- Model files missing or incorrect\n- Players not clearly visible in frames\n- Video quality issues")
                        except Exception as e:
                            st.error(f"Error tracking players: {str(e)}")
                            if DEBUG:
                                st.error(traceback.format_exc())
                    
                    # Get unique players for this event
                    players = st.session_state.player_tracker.get_unique_players_for_event(selected_event)
                    
                    if not players:
                        st.warning("No players detected in this event.")
                        
                        # Debug info to help troubleshoot
                        st.write("Debug information:")
                        if selected_event in st.session_state.player_tracker.event_player_tracks:
                            tracks = st.session_state.player_tracker.event_player_tracks[selected_event]
                            st.write(f"- Tracking data available with {len(tracks)} frames")
                            
                            # Show sample frame from tracking
                            if tracks and len(tracks) > 0 and 'frame_path' in tracks[0]:
                                frame_path = tracks[0]['frame_path']
                                if os.path.exists(frame_path):
                                    st.write("Sample tracked frame:")
                                    st.image(frame_path)
                                else:
                                    st.write(f"- Frame file not found: {frame_path}")
                            
                            # Check if any frames have track data
                            frames_with_tracks = 0
                            for frame in tracks:
                                if 'tracks' in frame and frame['tracks']:
                                    frames_with_tracks += 1
                            st.write(f"- Frames with player detections: {frames_with_tracks}/{len(tracks)}")
                            
                        else:
                            st.write("- No tracking data available for this event")
                    else:
                        # Get event tracks for calculating player presence
                        event_tracks = st.session_state.player_tracker.event_player_tracks.get(selected_event, [])
                        if event_tracks:
                            # Calculate minimum required frames for 30% presence
                            total_frames = len(event_tracks)
                            min_required_frames = int(total_frames * 0.3)
                            
                            # Filter players by frame presence
                            filtered_players = []
                            player_presence = {}  # Store presence percentage for display
                            
                            for player in players:
                                global_id = player['global_id']
                                frame_count = 0
                                
                                # Count frames where this player appears
                                for frame_data in event_tracks:
                                    for track in frame_data.get('tracks', []):
                                        if track.get('global_id') == global_id:
                                            frame_count += 1
                                            break
                                
                                # Calculate presence percentage
                                presence_percentage = (frame_count / total_frames * 100) if total_frames > 0 else 0
                                player_presence[global_id] = presence_percentage
                                
                                # Only include players present in at least 30% of frames
                                if frame_count >= min_required_frames:
                                    # Add presence information to player info
                                    player['presence'] = presence_percentage
                                    filtered_players.append(player)
                            
                            # Use filtered_players instead of players for display
                            if filtered_players:
                                st.subheader(f"Key Players in this Event ({len(filtered_players)})")
                                st.write("Only showing players present in at least 30% of frames. Click on a player to track their movement:")
                                
                                # Calculate number of columns (up to 4 players per row)
                                cols_per_row = min(4, len(filtered_players))
                                num_rows = (len(filtered_players) + cols_per_row - 1) // cols_per_row
                                
                                # Display players in grid layout
                                for row in range(num_rows):
                                    cols = st.columns(cols_per_row)
                                    for col_idx in range(cols_per_row):
                                        player_idx = row * cols_per_row + col_idx
                                        if player_idx < len(filtered_players):
                                            player = filtered_players[player_idx]
                                            with cols[col_idx]:
                                                # Get player info
                                                global_id = player['global_id']
                                                jersey_num = player.get('jersey_num', "Unknown")
                                                label = f"Player #{jersey_num}" if jersey_num and jersey_num != "Unknown" else f"Player {global_id}"
                                                
                                                # Add presence percentage to label
                                                presence = player.get('presence', 0)
                                                label = f"{label} ({presence:.1f}%)"
                                                
                                                # Add CSS class for selected player
                                                selected_class = " selected-player" if st.session_state.selected_player == global_id else ""
                                                
                                                # Show thumbnail and make it clickable
                                                st.markdown(f'<div class="player-card{selected_class}">', unsafe_allow_html=True)
                                                
                                                if player.get('thumbnail') and os.path.exists(player['thumbnail']):
                                                    st.image(player['thumbnail'], caption=label, width=150)
                                                else:
                                                    # Placeholder image if no thumbnail available
                                                    st.markdown(f"📷 {label}")
                                                
                                                if st.button(f"Track Player", key=f"player_{global_id}"):
                                                    st.session_state.selected_player = global_id
                                                    st.rerun()
                                                
                                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.warning("No players were present in at least 30% of the frames for this event.")
                                
                                # Optionally show all players with their presence percentage
                                with st.expander("Show all detected players"):
                                    st.write("Players detected (with frame presence percentage):")
                                    for player in players:
                                        global_id = player['global_id']
                                        jersey_num = player.get('jersey_num', "Unknown")
                                        label = f"Player #{jersey_num}" if jersey_num and jersey_num != "Unknown" else f"Player {global_id}"
                                        presence = player_presence.get(global_id, 0)
                                        st.write(f"{label}: {presence:.1f}% of frames")
                        else:
                            st.warning("No tracking data available for this event.")
                            
                        # Show player tracking if a player is selected
                        if st.session_state.selected_player:
                            st.subheader(f"Tracking Player {st.session_state.selected_player}")
                            
                            # Find player info
                            selected_player_info = None
                            for p in players:
                                if p['global_id'] == st.session_state.selected_player:
                                    selected_player_info = p
                                    break
                            
                            if selected_player_info:
                                jersey_num = selected_player_info.get('jersey_num', "Unknown")
                                label = f"Player #{jersey_num}" if jersey_num and jersey_num != "Unknown" else f"Player {st.session_state.selected_player}"
                                
                                # Create a container for the tracking visualization
                                video_container = st.container()
                                
                                with video_container:
                                    with st.spinner(f"Creating tracking visualization for {label}..."):
                                        # Get tracking data for this player in this event
                                        event_tracks = st.session_state.player_tracker.event_player_tracks.get(selected_event, [])
                                        
                                        if not event_tracks:
                                            st.error("No tracking data available for this event.")
                                        else:
                                            # Check if we need to create or if we already have a tracking video
                                            tracking_video_path = os.path.join(
                                                st.session_state.player_tracker.temp_dir, 
                                                f"player_{st.session_state.selected_player}_tracking.mp4"
                                            )
                                            
                                            # Try to create the tracking video
                                            try:
                                                # Show debug info
                                                st.info(f"Creating tracking visualization using {len(event_tracks)} frames")
                                                
                                                # Create tracking video
                                                new_video_path = st.session_state.player_tracker.create_player_tracking_video(
                                                    selected_event,
                                                    st.session_state.selected_player,
                                                    tracking_video_path
                                                )
                                                
                                                # Check if video was created successfully
                                                if new_video_path and os.path.exists(new_video_path) and os.path.getsize(new_video_path) > 0:
                                                    # Success - show the video
                                                    st.success("Tracking video created successfully!")
                                                    with open(new_video_path, "rb") as f:
                                                        video_bytes = f.read()
                                                    st.video(video_bytes)
                                                    
                                                    # Add button to download tracking video
                                                    with open(new_video_path, "rb") as file:
                                                        st.download_button(
                                                            label="Download Tracking Video",
                                                            data=file,
                                                            file_name=f"{label}_tracking.mp4",
                                                            mime="video/mp4"
                                                        )
                                                else:
                                                    # Video creation failed - show individual frames instead
                                                    st.warning("Could not create tracking video. Showing individual frames instead.")
                                                    
                                                    # Find frames with this player
                                                    frames_with_player = []
                                                    for frame_idx, frame_data in enumerate(event_tracks):
                                                        player_found = False
                                                        for track in frame_data.get('tracks', []):
                                                            if track.get('global_id') == st.session_state.selected_player:
                                                                player_found = True
                                                                break
                                                        
                                                        if player_found and 'frame_path' in frame_data:
                                                            frames_with_player.append(frame_data['frame_path'])
                                                    
                                                    if frames_with_player:
                                                        st.write(f"Found player in {len(frames_with_player)} frames")
                                                        
                                                        # Show a limited number of frames (max 9)
                                                        frames_to_show = frames_with_player[:20]
                                                        cols_per_row = 3
                                                        
                                                        for i in range(0, len(frames_to_show), cols_per_row):
                                                            cols = st.columns(cols_per_row)
                                                            for j in range(cols_per_row):
                                                                idx = i + j
                                                                if idx < len(frames_to_show):
                                                                    frame_path = frames_to_show[idx]
                                                                    if os.path.exists(frame_path):
                                                                        cols[j].image(frame_path, caption=f"Frame {idx+1}")
                                                    else:
                                                        st.error("Player not found in any frames.")
                                            
                                            except Exception as e:
                                                st.error(f"Error creating tracking visualization: {str(e)}")
                                                
                                                # Show the error details
                                                with st.expander("Show error details"):
                                                    st.code(traceback.format_exc())
                                                
                                                # Manual debug info
                                                st.write("Debug information:")
                                                st.write(f"- Player ID: {st.session_state.selected_player}")
                                                st.write(f"- Event key: {selected_event}")
                                                st.write(f"- Number of frames with tracking data: {len(event_tracks)}")
                                                
                                                # Show some sample frames if available
                                                sample_frames = []
                                                for frame_data in event_tracks:
                                                    if 'frame_path' in frame_data and os.path.exists(frame_data['frame_path']):
                                                        sample_frames.append(frame_data['frame_path'])
                                                        if len(sample_frames) >= 3:
                                                            break
                                                
                                                if sample_frames:
                                                    st.write("Sample frames from tracking:")
                                                    cols = st.columns(len(sample_frames))
                                                    for i, frame_path in enumerate(sample_frames):
                                                        cols[i].image(frame_path, caption=f"Sample frame {i+1}")
                            else:
                                st.error("Selected player information not found.")
        
        with highlights_tab:
            st.subheader("Player Highlights & Analysis")
            
            if not st.session_state.player_tracker or not hasattr(st.session_state.player_tracker, 'tracker'):
                st.info("Please analyze a video first and track players in the Player Tracking tab.")
            else:
                # Create directories for generated files if they don't exist
                os.makedirs("highlights", exist_ok=True)
                os.makedirs("trajectories", exist_ok=True)
                os.makedirs("heatmaps", exist_ok=True)
                
                # Determine our output directory for this video
                output_dir = None
                if hasattr(st.session_state.player_tracker, 'output_dir') and st.session_state.player_tracker.output_dir:
                    output_dir = st.session_state.player_tracker.output_dir
                    # Create subdirectories in the output directory
                    os.makedirs(os.path.join(output_dir, "highlight_clips"), exist_ok=True)
                    os.makedirs(os.path.join(output_dir, "trajectories"), exist_ok=True)
                    os.makedirs(os.path.join(output_dir, "heatmaps"), exist_ok=True)
                    st.info(f"Using tracking output directory: {output_dir}")
                
                # Scan for existing highlight videos and trajectory data
                if not st.session_state.player_highlights:
                    # First check if there's an existing highlight data JSON
                    highlight_data = None
                    
                    # Check the output directory from tracking first if available
                    highlight_data_paths = []
                    if output_dir:
                        highlight_data_paths.append(os.path.join(output_dir, "highlight_data.json"))
                    
                    # Add other standard locations
                    highlight_data_paths.extend([
                        "highlights/highlight_data.json",
                        "output/highlight_data.json"
                    ])
                    
                    for data_path in highlight_data_paths:
                        if os.path.exists(data_path):
                            try:
                                with open(data_path, 'r') as f:
                                    highlight_data = json.load(f)
                                st.success(f"Loaded existing highlight data from {data_path}")
                                break
                            except Exception as e:
                                st.warning(f"Error loading existing highlight data from {data_path}: {str(e)}")
                    
                    # If we didn't find an existing file, generate highlight data
                    if not highlight_data:
                        with st.spinner("Generating player highlights..."):
                            try:
                                # Determine where to save highlight data
                                highlight_output_path = "highlights/highlight_data.json"
                                if output_dir:
                                    highlight_output_path = os.path.join(output_dir, "highlight_data.json")
                                
                                # Generate highlight data for all players
                                highlight_data = st.session_state.player_tracker.tracker.generate_player_highlight_data(
                                    output_path=highlight_output_path,
                                    min_segment_duration=1.0,
                                    min_total_duration=2.0
                                )
                                
                                if highlight_data:
                                    st.success(f"Generated highlight data for {len(highlight_data)} players")
                            except Exception as e:
                                st.error(f"Error generating player highlights: {str(e)}")
                                if DEBUG:
                                    st.error(traceback.format_exc())
                    
                    # Now scan for existing highlight videos
                    if not highlight_data:
                        # If we couldn't get highlight data, scan for videos directly using the scan function
                        player_videos = scan_existing_highlight_videos(output_dir)
                        
                        # Create highlight data structure from the found videos
                        existing_players = {}
                        for player_id, video_path in player_videos.items():
                            existing_players[player_id] = {
                                'total_duration_sec': 10.0,  # Default duration
                                'segments': [{'duration_sec': 10.0}]  # Dummy segment
                            }
                        
                        # Use the discovered players
                        if existing_players:
                            highlight_data = existing_players
                            st.success(f"Found {len(existing_players)} players with existing highlight videos")
                    
                    # Store in session state if we found or generated data
                    if highlight_data:
                        st.session_state.player_highlights = highlight_data
                
                # If we have highlight data, process it
                if st.session_state.player_highlights:
                    # Get player data
                    all_players = st.session_state.player_highlights
                    
                    # Create a list of players, sorted by total duration
                    player_list = []
                    for key, player_data in all_players.items():
                        # Get jersey number if available
                        jersey_num = None
                        global_id = None
                        
                        if isinstance(key, int):
                            jersey_num = key
                        elif isinstance(key, str) and key.startswith("gid_"):
                            # Extract global ID
                            try:
                                global_id = int(key.replace("gid_", ""))
                                # Try to find jersey from SoccerTracker
                                if hasattr(st.session_state.player_tracker.tracker, 'reid'):
                                    jersey_num, _ = st.session_state.player_tracker.tracker.reid.get_best_jersey(global_id)
                            except:
                                pass
                        
                        # Check for existing highlight videos using the find function with output directory
                        highlight_video_path = find_highlight_video_for_player(key, output_dir)
                        
                        # Check for existing trajectory in output directory first, then standard locations
                        trajectory_path = None
                        if global_id:
                            # Check output directory first if available
                            if output_dir:
                                output_traj_path = os.path.join(output_dir, "trajectories", f"player_{global_id}_trajectory.jpg")
                                if os.path.exists(output_traj_path) and os.path.getsize(output_traj_path) > 0:
                                    trajectory_path = output_traj_path
                            
                            # Then check standard locations
                            if not trajectory_path:
                                standard_traj_paths = [
                                    f"trajectories/player_{global_id}_trajectory.jpg",
                                    f"output/player_{global_id}_trajectory.jpg"
                                ]
                                
                                for path in standard_traj_paths:
                                    if os.path.exists(path) and os.path.getsize(path) > 0:
                                        trajectory_path = path
                                        break
                        
                        # Check for existing heatmap
                        heatmap_path = None
                        if global_id:
                            # Check output directory first if available
                            if output_dir:
                                output_heatmap_path = os.path.join(output_dir, "heatmaps", f"player_{global_id}_heatmap.jpg")
                                if os.path.exists(output_heatmap_path) and os.path.getsize(output_heatmap_path) > 0:
                                    heatmap_path = output_heatmap_path
                            
                            # Then check standard locations
                            if not heatmap_path:
                                standard_heatmap_paths = [
                                    f"heatmaps/player_{global_id}_heatmap.jpg",
                                    f"output/player_{global_id}_heatmap.jpg"
                                ]
                                
                                for path in standard_heatmap_paths:
                                    if os.path.exists(path) and os.path.getsize(path) > 0:
                                        heatmap_path = path
                                        break
                        
                        # Add to player list
                        player_list.append({
                            'key': key,
                            'global_id': global_id,
                            'jersey_num': jersey_num,
                            'total_duration': player_data.get('total_duration_sec', 0),
                            'segments': player_data.get('segments', []),
                            'display_name': f"Player #{jersey_num}" if jersey_num else f"Player {key}",
                            'highlight_video_path': highlight_video_path,
                            'trajectory_path': trajectory_path,
                            'heatmap_path': heatmap_path
                        })
                    
                    # Sort by total duration (longest first)
                    player_list.sort(key=lambda x: x['total_duration'], reverse=True)
                    
                    # Check if we're in player detail view or player grid view
                    if st.session_state.selected_player and st.session_state.selected_player in all_players:
                        # PLAYER DETAIL VIEW
                        
                        # Find the player in our list
                        selected_player = None
                        for player in player_list:
                            if player['key'] == st.session_state.selected_player:
                                selected_player = player
                                break
                        
                        if selected_player:
                            # Get player data from session state or use the one we found
                            player_data = st.session_state.player_data.get(st.session_state.selected_player, {})
                            
                            # Update with any path we found during scanning (in case session data is outdated)
                            if selected_player.get('trajectory_path') and not player_data.get('trajectory_path'):
                                player_data['trajectory_path'] = selected_player['trajectory_path']
                            
                            if selected_player.get('heatmap_path') and not player_data.get('heatmap_path'):
                                player_data['heatmap_path'] = selected_player['heatmap_path']
                                
                            if selected_player.get('highlight_video_path') and not player_data.get('highlight_video_path'):
                                player_data['highlight_video_path'] = selected_player['highlight_video_path']
                            
                            # Back button at the top
                            if st.button("← Back to Player Gallery", type="primary"):
                                st.session_state.selected_player = None
                                st.rerun()
                            
                            # Player header with key info
                            st.markdown(f"## {selected_player['display_name']}")
                            
                            # Create tabs for different player analysis views
                            player_tabs = st.tabs(["Highlights", "Profile", "Movement Analysis"])
                            highlights_tab, profile_tab, analysis_tab = player_tabs
                            
                            # 1. HIGHLIGHTS TAB
                            with highlights_tab:
                                # Highlight Stats
                                st.subheader("Highlight Statistics")
                                player_info = all_players[st.session_state.selected_player]
                                segments = player_info.get('segments', [])
                                
                                # Create stats grid
                                stat_col1, stat_col2, stat_col3 = st.columns(3)
                                with stat_col1:
                                    st.metric("Total Highlight Duration", f"{player_info.get('total_duration_sec', 0):.1f}s")
                                with stat_col2:
                                    st.metric("Highlight Segments", f"{len(segments)}")
                                with stat_col3:
                                    avg_duration = player_info.get('average_segment_duration', 0)
                                    if not avg_duration and segments:
                                        avg_duration = sum(s.get('duration_sec', 0) for s in segments) / len(segments)
                                    st.metric("Avg. Segment Duration", f"{avg_duration:.1f}s")
                                
                                # Highlight video section
                                st.subheader("Player Highlight Video")
                                
                                # Check if we already have a highlight video
                                highlight_video_path = player_data.get('highlight_video_path')
                                
                                if not highlight_video_path:
                                    # Use the function to find highlight videos with output directory
                                    highlight_video_path = find_highlight_video_for_player(st.session_state.selected_player, output_dir)
                                    
                                    # Update player data with the found path
                                    if highlight_video_path:
                                        player_data['highlight_video_path'] = highlight_video_path
                                        st.session_state.player_data[st.session_state.selected_player]['highlight_video_path'] = highlight_video_path
                                
                                # Show generate button or video
                                if not highlight_video_path:
                                    st.info("No highlight video found for this player.")
                                    
                                    # Create directories if they don't exist
                                    highlight_dir = "highlights/videos"
                                    if output_dir:
                                        highlight_dir = os.path.join(output_dir, "highlight_clips")
                                    os.makedirs(highlight_dir, exist_ok=True)
                                    
                                    if st.button("Generate Highlight Video", type="primary"):
                                        with st.spinner("Generating player highlight video..."):
                                            try:
                                                # Use the existing function from the detect_track_v3.py module
                                                from detect_track import extract_player_highlights
                                                
                                                # Determine highlight data path
                                                highlight_data_path = "highlights/highlight_data.json"
                                                if output_dir:
                                                    output_highlight_path = os.path.join(output_dir, "highlight_data.json")
                                                    if os.path.exists(output_highlight_path):
                                                        highlight_data_path = output_highlight_path
                                                
                                                # Call the extract_player_highlights function
                                                highlight_clips = extract_player_highlights(
                                                    video_path="temp_video.mp4",
                                                    highlight_data_path=highlight_data_path,
                                                    output_dir=highlight_dir,
                                                    selected_identifiers=[st.session_state.selected_player],
                                                    max_clips=5,
                                                    highlight_seconds=2.0
                                                )
                                                
                                                # Check if we got some clips back
                                                if highlight_clips and st.session_state.selected_player in highlight_clips:
                                                    clip_paths = highlight_clips[st.session_state.selected_player]
                                                    if clip_paths:
                                                        # Store the first clip path
                                                        player_data['highlight_video_path'] = clip_paths[0]
                                                        st.session_state.player_data[st.session_state.selected_player]['highlight_video_path'] = clip_paths[0]
                                                        st.success(f"Generated {len(clip_paths)} highlight clips!")
                                                        st.rerun()  # Refresh to show the video
                                                else:
                                                    st.error("No highlight clips were generated")
                                            except Exception as e:
                                                st.error(f"Error generating highlight video: {str(e)}")
                                                if DEBUG:
                                                    st.error(traceback.format_exc())
                                else:
                                    # Display highlight video
                                    try:
                                        # First read the video file into a BytesIO object
                                        video_bytes = io.BytesIO()
                                        with open(highlight_video_path, "rb") as f:
                                            video_bytes.write(f.read())
                                        
                                        # Reset the position to the beginning of the buffer
                                        video_bytes.seek(0)
                                        
                                        # Save to a temporary file
                                        temp_video_path = os.path.join(tempfile.gettempdir(), f"temp_highlight_{uuid.uuid4()}.mp4")
                                        with open(temp_video_path, "wb") as f:
                                            f.write(video_bytes.getbuffer())
                                        
                                        # Display the video
                                        st.video(temp_video_path)
                                        
                                        # Add download button
                                        with open(highlight_video_path, "rb") as file:
                                            st.download_button(
                                                label="Download Highlight Video",
                                                data=file,
                                                file_name=f"player_{st.session_state.selected_player}_highlights.mp4",
                                                mime="video/mp4"
                                            )
                                    except Exception as e:
                                        st.error(f"Error displaying highlight video: {str(e)}")
                                        if os.path.exists(highlight_video_path):
                                            st.error(f"Video file exists but cannot be played: {highlight_video_path}")
                                
                                # Individual highlight segments section
                                if segments:
                                    with st.expander("View Individual Highlight Segments"):
                                        st.write(f"Showing top {min(5, len(segments))} segments sorted by duration")
                                        
                                        # Sort segments by duration (longest first)
                                        sorted_segments = sorted(segments, key=lambda s: s.get('duration_sec', 0), reverse=True)
                                        
                                        # Show top 5 segments
                                        for i, segment in enumerate(sorted_segments[:5]):
                                            # Create expandable section for each segment
                                            duration = segment.get('duration_sec', 0)
                                            start_frame = segment.get('start_frame', 0)
                                            end_frame = segment.get('end_frame', 0)
                                            
                                            st.markdown(f"#### Segment {i+1}")
                                            st.markdown(f"Duration: {duration:.1f}s (Frames {start_frame}-{end_frame})")
                                            
                                            # Show stats for this segment
                                            visible_frames = segment.get('visible_frames', 0)
                                            jersey_frames = segment.get('jersey_detected_frames', 0)
                                            jersey_visibility = segment.get('jersey_visibility_perc', 0)
                                            
                                            s_col1, s_col2, s_col3 = st.columns(3)
                                            with s_col1:
                                                st.metric("Visible Frames", visible_frames)
                                            with s_col2:
                                                st.metric("Jersey Detected", jersey_frames)
                                            with s_col3:
                                                st.metric("Jersey Visibility", f"{jersey_visibility:.1f}%")
                                            
                                            if i < len(sorted_segments) - 1:
                                                st.markdown("---")
                            
                            # 2. PROFILE TAB
                            with profile_tab:
                                st.subheader("Player Information")
                                
                                # Player information form
                                with st.form(key=f"player_form_{st.session_state.selected_player}"):
                                    player_name = st.text_input("Player Name", value=player_data.get('name', selected_player['display_name']))
                                    jersey_number = st.text_input("Jersey Number", value=player_data.get('jersey', selected_player['jersey_num'] if selected_player['jersey_num'] else ''))
                                    team = st.selectbox("Team", ["Unassigned", "Team A", "Team B"], index=["Unassigned", "Team A", "Team B"].index(player_data.get('team', 'Unassigned')))
                                    notes = st.text_area("Notes", value=player_data.get('notes', ''))
                                    
                                    submit = st.form_submit_button("Save Player Information")
                                    if submit:
                                        # Update player data
                                        st.session_state.player_data[st.session_state.selected_player] = {
                                            'name': player_name,
                                            'jersey': jersey_number,
                                            'team': team,
                                            'notes': notes,
                                            'trajectory_path': player_data.get('trajectory_path'),
                                            'highlight_video_path': player_data.get('highlight_video_path'),
                                            'heatmap_path': player_data.get('heatmap_path')
                                        }
                                        st.success("Player information updated!")
                            
                            # 3. MOVEMENT ANALYSIS TAB
                            with analysis_tab:
                                st.subheader("Player Movement Analysis")
                                
                                # Get global ID for movement analysis
                                global_id = selected_player.get('global_id')
                                
                                if not global_id:
                                    st.warning("Movement analysis requires player tracking data with a global ID.")
                                else:
                                    # Create tabs for different visualizations
                                    viz_tabs = st.tabs(["Trajectory", "Heatmap"])
                                    traj_tab, heat_tab = viz_tabs
                                    
                                    # TRAJECTORY TAB
                                    with traj_tab:
                                        st.subheader("Player Trajectory")
                                        
                                        # Check if trajectory exists
                                        if not player_data.get('trajectory_path'):
                                            st.info("No trajectory visualization available for this player.")
                                            
                                            # Generate trajectory button
                                            if st.button("Generate Player Trajectory", type="primary"):
                                                with st.spinner("Generating trajectory visualization..."):
                                                    # Determine output path
                                                    output_traj_path = f"trajectories/player_{global_id}_trajectory.jpg"
                                                    if output_dir:
                                                        traj_dir = os.path.join(output_dir, "trajectories")
                                                        os.makedirs(traj_dir, exist_ok=True)
                                                        output_traj_path = os.path.join(traj_dir, f"player_{global_id}_trajectory.jpg")
                                                    
                                                    # Generate trajectory for this player
                                                    trajectory_path = st.session_state.player_tracker.generate_player_trajectory(
                                                        global_id,
                                                        output_traj_path
                                                    )
                                                    
                                                    if trajectory_path and os.path.exists(trajectory_path):
                                                        # Update player data
                                                        player_data['trajectory_path'] = trajectory_path
                                                        st.session_state.player_data[st.session_state.selected_player]['trajectory_path'] = trajectory_path
                                                        st.success("Trajectory generated successfully!")
                                                        st.rerun()
                                        else:
                                            # Display existing trajectory
                                            st.image(player_data['trajectory_path'], caption="Player movement trajectory")
                                            
                                            # Add download button
                                            with open(player_data['trajectory_path'], "rb") as file:
                                                st.download_button(
                                                    label="Download Trajectory Image",
                                                    data=file,
                                                    file_name=f"player_{st.session_state.selected_player}_trajectory.jpg",
                                                    mime="image/jpeg"
                                                )
                                    
                                    # HEATMAP TAB
                                    with heat_tab:
                                        st.subheader("Player Heatmap")
                                        
                                        # Check if heatmap exists
                                        if not player_data.get('heatmap_path'):
                                            st.info("No heatmap visualization available for this player.")
                                            
                                            # Generate heatmap button
                                            if st.button("Generate Player Heatmap", type="primary"):
                                                with st.spinner("Generating heatmap visualization..."):
                                                    # Determine output path
                                                    output_heatmap_path = f"heatmaps/player_{global_id}_heatmap.jpg"
                                                    if output_dir:
                                                        heatmap_dir = os.path.join(output_dir, "heatmaps")
                                                        os.makedirs(heatmap_dir, exist_ok=True)
                                                        output_heatmap_path = os.path.join(heatmap_dir, f"player_{global_id}_heatmap.jpg")
                                                    
                                                    # Generate heatmap for this player
                                                    heatmap_path = st.session_state.player_tracker.generate_player_heatmap(
                                                        global_id,
                                                        output_heatmap_path
                                                    )
                                                    
                                                    if heatmap_path and os.path.exists(heatmap_path):
                                                        # Update player data
                                                        player_data['heatmap_path'] = heatmap_path
                                                        st.session_state.player_data[st.session_state.selected_player]['heatmap_path'] = heatmap_path
                                                        st.success("Heatmap generated successfully!")
                                                        st.rerun()
                                        else:
                                            # Display existing heatmap
                                            st.image(player_data['heatmap_path'], caption="Player movement heatmap")
                                            
                                            # Add download button
                                            with open(player_data['heatmap_path'], "rb") as file:
                                                st.download_button(
                                                    label="Download Heatmap Image",
                                                    data=file,
                                                    file_name=f"player_{st.session_state.selected_player}_heatmap.jpg",
                                                    mime="image/jpeg"
                                                )
                        
                        else:
                            st.error("Selected player information not found.")
                            if st.button("Return to Player Gallery"):
                                st.session_state.selected_player = None
                                st.rerun()
                    
                    else:
                        # PLAYER GRID VIEW
                        
                        # Separate top 22 players and others
                        top_players = player_list[:22] if len(player_list) > 22 else player_list
                        other_players = player_list[22:] if len(player_list) > 22 else []
                        
                        # Search box for filtering players
                        search_query = st.text_input("Search players by jersey number or ID", "")
                        
                        if search_query:
                            filtered_players = []
                            for player in player_list:
                                # Check jersey number
                                if player['jersey_num'] and str(player['jersey_num']) in search_query:
                                    filtered_players.append(player)
                                # Check player ID
                                elif str(player['key']) in search_query:
                                    filtered_players.append(player)
                                # Check display name
                                elif search_query.lower() in player['display_name'].lower():
                                    filtered_players.append(player)
                            
                            if filtered_players:
                                st.success(f"Found {len(filtered_players)} matching players")
                                top_players = filtered_players
                                other_players = []
                            else:
                                st.warning(f"No players found matching '{search_query}'")
                        
                        # Player grid header
                        if not search_query:
                            st.markdown("### Featured Players")
                            st.caption(f"Showing top {len(top_players)} players with the most highlight content")
                        
                        # Initialize player data if empty
                        if not st.session_state.player_data:
                            for player in player_list:
                                player_key = player['key']
                                st.session_state.player_data[player_key] = {
                                    'name': player['display_name'],
                                    'jersey': player['jersey_num'] if player['jersey_num'] else '',
                                    'team': 'Unassigned',
                                    'notes': '',
                                    'trajectory_path': player['trajectory_path'],
                                    'highlight_video_path': player['highlight_video_path'],
                                    'heatmap_path': player['heatmap_path']
                                }
                        
                        # Create player grid with consistent width cards
                        cols_per_row = 4
                        
                        # Function to render player card
                        def render_player_card(player):
                            player_key = player['key']
                            
                            # Create container for the card
                            with st.container():
                                # Show available resources with icons
                                has_highlight = "🎬" if player['highlight_video_path'] else "❌"
                                has_trajectory = "🗺️" if player['trajectory_path'] else "❌"
                                has_heatmap = "🔥" if player['heatmap_path'] else "❌"
                                
                                # Card with light border
                                st.markdown(f"""
                                <div style="padding:10px; border:1px solid #eee; border-radius:5px; margin-bottom:10px;">
                                    <h4 style="margin-top:0;">{player['display_name']}</h4>
                                    <p>Highlight Time: {player['total_duration']:.1f}s</p>
                                    <p>Highlights: {has_highlight} | Trajectory: {has_trajectory} | Heatmap: {has_heatmap}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Button to view player details
                                if st.button(f"View Details", key=f"details_{player_key}"):
                                    st.session_state.selected_player = player_key
                                    st.rerun()
                        
                        # Render player grid
                        for i in range(0, len(top_players), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j in range(cols_per_row):
                                idx = i + j
                                if idx < len(top_players):
                                    with cols[j]:
                                        render_player_card(top_players[idx])
                        
                        # Display "Other Players" section if there are additional players
                        if other_players and not search_query:
                            with st.expander(f"View {len(other_players)} Additional Players"):
                                st.caption("Players with less highlight content")
                                
                                # Create grid layout for other players
                                for i in range(0, len(other_players), cols_per_row):
                                    cols = st.columns(cols_per_row)
                                    for j in range(cols_per_row):
                                        idx = i + j
                                        if idx < len(other_players):
                                            with cols[j]:
                                                render_player_card(other_players[idx])
                        
                        # Add helpful instructions at the bottom
                        st.markdown("---")
                        st.caption("Click on a player card to view details, highlights, and movement analysis.")
                        st.caption("Player highlights are generated from tracked events in the match video.")
        # Player Stats Tab
        with stats_tab:
            st.info("Player statistics feature is coming soon!")
            
            # Placeholder content for future implementation
            st.subheader("Player Performance Statistics")
            
            if st.session_state.player_tracker and hasattr(st.session_state.player_tracker, 'tracker'):
                # Show overview of all tracked players
                st.write("This tab will provide detailed statistics for each player:")
                st.markdown("""
                - Movement heat maps
                - Speed and acceleration analysis
                - Event participation (goals, assists, etc.)
                - Interaction networks with other players
                - Performance metrics over time
                """)
                
                # Show some placeholder charts
                st.subheader("Player Performance Metrics (Coming Soon)")
                cols = st.columns(2)
                
                with cols[0]:
                    # Placeholder for a chart
                    chart_placeholder = st.empty()
                    chart_placeholder.markdown("""
                    ```
                    +-------------------------+
                    |                         |
                    |     Player Speed        |
                    |     Chart Coming        |
                    |     Soon                |
                    |                         |
                    +-------------------------+
                    ```
                    """)
                
                with cols[1]:
                    # Placeholder for another chart
                    chart_placeholder = st.empty()
                    chart_placeholder.markdown("""
                    ```
                    +-------------------------+
                    |                         |
                    |     Player Heatmap      |
                    |     Chart Coming        |
                    |     Soon                |
                    |                         |
                    +-------------------------+
                    ```
                    """)
            else:
                st.warning("Please analyze a video and track players to enable statistics.")
    
    # Team Analysis Tab
    else:  # app_mode == "Team Analysis"
        with team_tab:
            st.info("Team analysis features are under development.")
            
            st.subheader("Team Performance Overview")
            
            # Placeholder content
            st.markdown("""
            This section will provide team-level analysis including:
            - Team formation and positioning
            - Passing networks
            - Team movement patterns
            - Possession statistics
            - Team heat maps
            """)
            
            # Show placeholder image
            st.markdown("""
            ```
            +------------------------------------------+
            |                                          |
            |        Team Formation Analysis           |
            |        Coming Soon                       |
            |                                          |
            |                                          |
            |                                          |
            +------------------------------------------+
            ```
            """)
        
        with tactical_tab:
            st.info("Tactical analysis features are under development.")
            
            st.subheader("Match Tactical Breakdown")
            
            # Placeholder content
            st.markdown("""
            This section will provide tactical insights including:
            - Formation analysis
            - Pressing patterns
            - Defensive organization
            - Attacking strategies
            - Set pieces analysis
            """)
            
            # Show placeholder image
            st.markdown("""
            ```
            +------------------------------------------+
            |                                          |
            |        Tactical Heatmap                  |
            |        Coming Soon                       |
            |                                          |
            |                                          |
            |                                          |
            +------------------------------------------+
            ```
            """)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Football Analytics Platform")
    parser.add_argument("--video", type=str, help="Path to football match video")
    parser.add_argument("--output", type=str, help="Path to output file (optional)")
    
    args = parser.parse_args()
    
    if args.video:
        events, analyzer = analyze_football_video(args.video, args.output)
        analyzer.cleanup()
    else:
        # If no command line args, run as Streamlit app
        try:
            streamlit_app()
        except Exception as e:
            if DEBUG:
                traceback.print_exc()
            print(f"Error: {str(e)}")
            print("To run as CLI: python script.py --video path/to/video.mp4")
            print("To run as Streamlit app: streamlit run script.py")