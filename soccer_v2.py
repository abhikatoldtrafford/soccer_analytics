import os
import cv2
import base64
import time
import json
import numpy as np
import tempfile
from datetime import timedelta
from tqdm import tqdm
from openai import OpenAI
import streamlit as st
from collections import defaultdict
import threading
from PIL import Image
import io
import traceback

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
WINDOW_SIZE = 30         # Seconds per analysis window
WINDOW_STEP = 25          # Seconds to advance window between analyses
FRAME_INTERVAL = 1       # Seconds between frames
SEQUENCE_SIZE = 30       # Number of consecutive frames to analyze at once
MODEL = "gpt-4o-mini"    # Model to use for analysis
RESIZE_DIMENSION = 800   # Resize image dimension for API

# Debug mode - set to True to print more information
DEBUG = True

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
        """Create a prompt for analyzing player performance from frames"""
        return """You are an expert football analyst identifying key player performance moments from sequences of consecutive frames.

I'll show you a sequence of consecutive frames from a football match. Each frame is timestamped.

Focus ONLY on identifying these football actions that demonstrate player skill and performance:

1. GOAL
   - Definition: Ball completely crosses goal line between posts and under crossbar
   - Visual indicators:
     * Shot being taken, ball crossing line/entering net
     * Player celebrations with raised arms
     * Goalkeeper retrieving ball from net

2. GOOD PASS
   - Definition: An accurate pass that finds a teammate in space or breaks defensive lines
   - Visual indicators:
     * Long-range pass that changes field position
     * Pass that eliminates defenders and creates advantage
     * Pass that leads to attacking opportunity
     * Pass showing excellent vision and technique

3. THROUGH BALL
   - Definition: Pass played into space behind the defense for an attacker to run onto
   - Visual indicators:
     * Ball played between or behind defenders
     * Attacking player running onto ball in space
     * Defense caught out of position

4. SHOT ON TARGET
   - Definition: Any attempt that would enter the goal if not saved
   - Visual indicators:
     * Player striking ball toward goal
     * Goalkeeper making save
     * Ball trajectory heading toward goal frame

5. GOOD GOALKEEPING
   - Definition: Goalkeeper making difficult saves or commanding their area well
   - Visual indicators:
     * Diving save
     * Coming out to claim crosses
     * One-on-one saves
     * Good positioning and shot-stopping

6. SKILL MOVE
   - Definition: Technical skill to beat an opponent
   - Visual indicators:
     * Dribbling past defenders
     * Feints, step-overs, or clever touches
     * Creating space with technical ability

IMPORTANT GUIDELINES:
- ONLY identify these 6 specific events related to player performance
- Look at the chronological sequence to understand the flow of play
- Provide the EXACT time range for the event based on the frame timestamps
- Be very specific about what visual evidence you see and which player(s) performed well
- If you're uncertain about an event, DO NOT include it

Return your findings in this EXACT format:
[MM:SS-MM:SS] EVENT_TYPE: Brief description with specific visual evidence of player performance

Example:
[35:20-35:28] GOOD PASS: #10 delivers precision long ball from midfield that splits defenders, allowing teammate to control in dangerous position

If no clear performance events in these frames, respond with "No key events detected in this sequence."
"""

    def _analyze_frames(self, frames_data, window_idx, total_windows, event_callback=None):
        """Analyze frames using OpenAI API"""
        window_events = []
        
        # Skip if not enough frames
        if len(frames_data) < SEQUENCE_SIZE:
            return window_events
        
        # Process frames in sequence
        print(f"Processing {len(frames_data)} frames...")
        
        # Get first and last timestamp for the window
        start_time = frames_data[0]['timestamp']
        end_time = frames_data[-1]['timestamp']
        start_time_str = self._format_time(start_time)
        end_time_str = self._format_time(end_time)
        
        # Create message content
        message_content = [
            {"type": "text", "text": f"""
            Analyze these player performance moments from a football match.
            
            Time range: {start_time_str} to {end_time_str}
            
            The frames are in chronological order.
            Each frame has its timestamp displayed.
            
            Identify if any of these SPECIFIC player performance events are occurring:
            - GOAL
            - GOOD PASS
            - THROUGH BALL
            - SHOT ON TARGET
            - GOOD GOALKEEPING
            - SKILL MOVE
            
            ONLY report events you're absolutely certain about with precise visual evidence.
            """}
        ]
        
        # Add frames to message (using in-memory base64)
        # We'll use a subset of frames if there are too many
        selected_frames = frames_data
        if len(frames_data) > SEQUENCE_SIZE:
            # Select frames evenly distributed across the window
            indices = np.linspace(0, len(frames_data) - 1, SEQUENCE_SIZE, dtype=int)
            selected_frames = [frames_data[i] for i in indices]
        
        for frame_data in selected_frames:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_data['base64']}",
                    "detail": "high"
                }
            })
        
        # Define frame_ids for reference in results
        frame_ids = [frame_data['frame_id'] for frame_data in frames_data]
        
        # API call
        try:
            if DEBUG:
                print(f"Analyzing window {window_idx+1}/{total_windows}: {start_time_str} to {end_time_str} with {len(selected_frames)} frames")
            
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": self._create_player_analysis_prompt()},
                    {"role": "user", "content": message_content}
                ],
                temperature=0.2
            )
            
            event_text = response.choices[0].message.content
            
            if DEBUG:
                print(f"API Response: {event_text[:200]}...")
            
            # Parse events with time ranges
            events = self._parse_events(event_text, start_time, end_time, frame_ids)
            
            # Add events to window_events
            window_events.extend(events)
            
            # Call event_callback for live updates if provided
            if event_callback and events:
                if DEBUG:
                    print(f"Calling event_callback with {len(events)} events")
                event_callback(events)
            
            # Print events found in this window
            if events:
                print(f"  Window {window_idx+1}/{total_windows}: Found {len(events)} events")
                for event in events:
                    print(f"    [{self._format_time(event['start_time'])}-{self._format_time(event['end_time'])}] {event['event_type']}")
                    print(f"    Description: {event['description']}")
            
        except Exception as e:
            print(f"Error analyzing window {start_time_str}-{end_time_str}: {str(e)}")
            if DEBUG:
                traceback.print_exc()
            time.sleep(2)  # Wait before trying next window
        
        # Post-process: merge similar events that are close in time
        merged_events = self._merge_events(window_events)
        
        print(f"Window {window_idx+1}/{total_windows} complete: Found {len(merged_events)} distinct events")
        
        return merged_events

    def _parse_events(self, event_text, window_start_time, window_end_time, frame_ids):
        """Parse events from OpenAI response"""
        events = []
        
        if not event_text or "No key events detected" in event_text:
            return events
            
        lines = event_text.strip().split('\n')
        
        for line in lines:
            if '[' in line and ']' in line and ':' in line:
                # Extract time range
                time_range = line[line.find('[')+1:line.find(']')]
                
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
                    event_type, description = remaining.split(':', 1)
                    event_type = event_type.strip().upper()  # Normalize event type
                    description = description.strip()
                    
                    # Only accept the specific event types we're looking for
                    valid_events = ["GOAL", "GOOD PASS", "THROUGH BALL", "SHOT ON TARGET", "GOOD GOALKEEPING", "SKILL MOVE"]
                    
                    # Check for approximate matches too (e.g. "SHOT" matches "SHOT ON TARGET")
                    matched_event = None
                    for valid_event in valid_events:
                        if event_type == valid_event or valid_event in event_type:
                            matched_event = valid_event
                            break
                    
                    if matched_event:
                        events.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'event_type': matched_event,
                            'description': description,
                            'frame_ids': frame_ids
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
            
            formatted_events.append({
                'event_string': f"[{start_str}-{end_str}] {event['event_type']}: {event['description']}",
                'frame_ids': unique_frames,
                'start_time': event['start_time'],
                'end_time': event['end_time'],
                'event_type': event['event_type'],
                'description': event['description']
            })
        
        # Sort by start time
        formatted_events.sort(key=lambda x: x['start_time'])
        
        return formatted_events

    def analyze(self, progress_callback=None, event_callback=None):
        """Main analysis pipeline with callbacks for live updates"""
        try:
            # Analyze the video
            events = self.analyze_video(progress_callback, event_callback)
            return events
            
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

# Main function to use the analyzer
def analyze_football_video(video_path, output_path=None, progress_callback=None, event_callback=None):
    """Analyze football video and save results with callbacks for live updates"""
    if not output_path:
        output_path = os.path.splitext(video_path)[0] + "_events.txt"
    
    analyzer = FootballPlayerAnalyzer(video_path)
    events = analyzer.analyze(progress_callback, event_callback)
    
    # Save results
    with open(output_path, 'w') as f:
        f.write("FOOTBALL PLAYER PERFORMANCE EVENTS\n")
        f.write("=================================\n\n")
        for event in events:
            f.write(f"{event['event_string']}\n")
            f.write(f"  Frames: {', '.join(event['frame_ids'][:5])}...\n\n")
    
    print(f"\nResults saved to {output_path}")
    print("\nDetected Events:")
    for event in events:
        print(event['event_string'])
        print(f"  Frames: {', '.join(event['frame_ids'][:5])}...")
        print()
    
    return events, analyzer

# Enhanced Streamlit app function
def streamlit_app():
    st.set_page_config(layout="wide", page_title="Football Player Performance Analysis")
    
    st.title("Football Player Performance Analysis")
    st.write("Detects goals, good passes, through balls, shots on target, good goalkeeping, and skill moves")
    
    # Create columns for layout
    col1, col2 = st.columns([7, 3])
    
    # Global variables for event tracking
    if 'events' not in st.session_state:
        st.session_state.events = []
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
    if 'current_time' not in st.session_state:
        st.session_state.current_time = 0
    
    # Create a placeholder for the video player
    with col1:
        video_container = st.empty()
        time_indicator = st.empty()
    
    # Create placeholders for events display
    with col2:
        st.subheader("Detected Events (Live)")
        # Create a container to display events
        events_container = st.container()
        
        # Display events from session state
        if st.session_state.events:
            with events_container:
                for event in st.session_state.events:
                    st.write(event['event_string'])
    
    # File uploader for video
    uploaded_file = st.file_uploader("Upload a football match video", type=["mp4", "avi", "mov"])
    
    # Progress bar placeholder
    progress_bar = st.empty()
    status_text = st.empty()
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the video using Streamlit's native video player
        with col1:
            video_container.video(temp_video_path)
            time_indicator.write(f"Current position: {int(st.session_state.current_time // 60):02d}:{int(st.session_state.current_time % 60):02d}")
        
        # Create analyze button
        analyze_button = st.button("Start Player Performance Analysis", type="primary")
        
        if analyze_button:
            progress_value = progress_bar.progress(0)
            status_text.text("Starting analysis...")
            
            # Clear previous events
            st.session_state.events = []
            
            # Function to update the progress bar
            def update_progress(progress):
                progress_value.progress(progress)
            
            # Function to update events in real-time
            def update_events(new_events):
                # Convert new events to the formatted version
                formatted_events = []
                for event in new_events:
                    if 'event_string' not in event:
                        start_str = f"{int(event['start_time'] // 60):02d}:{int(event['start_time'] % 60):02d}"
                        end_str = f"{int(event['end_time'] // 60):02d}:{int(event['end_time'] % 60):02d}"
                        formatted_events.append({
                            'event_string': f"[{start_str}-{end_str}] {event['event_type']}: {event['description']}",
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
                
                # Force a rerun to display the new events
                # After testing, this had issues - let's use a more reliable approach
                # Instead, let's just display the events in the container
                with events_container:
                    for event in formatted_events:
                        st.write(event['event_string'])
                
                # Print to console for debugging
                if DEBUG:
                    print(f"Added {len(formatted_events)} new events, total: {len(st.session_state.events)}")
            
            try:
                # Run the analysis
                status_text.text("Analyzing video...")
                
                events, analyzer = analyze_football_video(
                    temp_video_path,
                    "player_performance.txt",
                    progress_callback=update_progress,
                    event_callback=update_events
                )
                
                # Store analyzer for frame access
                st.session_state.analyzer = analyzer
                
                # Update session state with final events
                if events and not st.session_state.events:
                    st.session_state.events = events
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Final events display
                with col2:
                    events_container.empty()
                    st.success(f"Detected {len(events)} player performance events")
                    
                    for idx, event in enumerate(events):
                        with st.expander(f"{event['event_string']}"):
                            st.write(event['event_string'])
                            
                            # Create video timestamp link
                            start_seconds = event['start_time']
                            st.write(f"Event time: {int(start_seconds // 60):02d}:{int(start_seconds % 60):02d}")
                            
                            # Display frames for this event (up to 3)
                            frames_to_show = event['frame_ids'][:3]
                            frame_cols = st.columns(len(frames_to_show))
                            for i, frame_id in enumerate(frames_to_show):
                                frame_path = analyzer.get_frame_path(frame_id)
                                if os.path.exists(frame_path):
                                    frame_cols[i].image(frame_path, caption=f"Frame {frame_id.replace('frame_', '')}")
                            
                            # Add button to seek to this event in the video
                            if st.button(f"View this event", key=f"view_{idx}"):
                                st.session_state.current_time = start_seconds
                                # Force video reload at the correct position
                                st.experimental_rerun()
                
                # Download button for results
                with open("player_performance.txt", "r") as f:
                    st.download_button(
                        label="Download Results",
                        data=f.read(),
                        file_name="player_performance.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                if DEBUG:
                    st.error(traceback.format_exc())
            
            # Clean up
            if os.path.exists(temp_video_path) and not st.session_state.events:
                os.remove(temp_video_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Football Player Performance Analysis")
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