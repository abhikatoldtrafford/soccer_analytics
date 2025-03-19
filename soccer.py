import os
import cv2
import base64
import time
import json
import numpy as np
from datetime import timedelta
from tqdm import tqdm
from openai import OpenAI
import streamlit as st
from collections import defaultdict

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
TEMP_DIR = "temp_files"
FRAME_INTERVAL = 1       # Seconds between frames
SEQUENCE_SIZE = 6        # Number of consecutive frames to analyze at once
SEQUENCE_OVERLAP = 2     # Number of frames to overlap between sequences
MODEL = "gpt-4o-mini"    # Using gpt-4o-mini as requested

class FootballAnalyzer:
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
        
        # Create directories for temporary files
        os.makedirs(TEMP_DIR, exist_ok=True)
        os.makedirs(f"{TEMP_DIR}/frames", exist_ok=True)
        
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Duration: {self._format_time(self.total_seconds)}")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"FPS: {self.fps:.2f}")
    
    def _format_time(self, seconds):
        """Format seconds to MM:SS format"""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def analyze_video(self):
        """Analyze the entire video in segments"""
        print("\nPerforming analysis of the entire video...")
        all_events = []
        
        # Divide the video into manageable segments (1-minute chunks)
        segment_duration = 60  # 1-minute chunks
        segments = [
            (i, min(i + segment_duration, self.total_seconds))
            for i in range(0, int(self.total_seconds), segment_duration)
        ]
        
        # Create progress bar for segments
        with tqdm(total=len(segments), desc="Segment Analysis") as segment_pbar:
            for segment_idx, (start_time, end_time) in enumerate(segments):
                print(f"\nAnalyzing segment {segment_idx+1}/{len(segments)}: {self._format_time(start_time)} - {self._format_time(end_time)}")
                
                # Extract frames for this segment
                frame_sequences = self._extract_frame_sequences(start_time, end_time)
                
                # Analyze frame sequences
                segment_events = self._analyze_frame_sequences(frame_sequences, segment_idx, len(segments))
                all_events.extend(segment_events)
                
                segment_pbar.update(1)
        
        return all_events

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
        
        with tqdm(total=(end_frame_pos - start_frame_pos) // frame_step, desc="Extracting Frames") as pbar:
            for frame_pos in range(start_frame_pos, end_frame_pos, frame_step):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Calculate timestamp
                timestamp = frame_pos / self.fps
                timestamp_str = self._format_time(timestamp)
                
                # Add timestamp to image
                # Add timestamp annotation
                frame_with_timestamp = frame.copy()
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
                
                # Save frame
                frame_filename = f"frame_{int(timestamp):06d}.jpg"
                frame_path = os.path.join(TEMP_DIR, "frames", frame_filename)
                cv2.imwrite(frame_path, frame_with_timestamp)
                
                frame_paths.append(frame_path)
                timestamps.append(timestamp)
                
                pbar.update(1)
        
        # Create sequences with overlap
        for i in range(0, len(frame_paths) - SEQUENCE_SIZE + 1, SEQUENCE_SIZE - SEQUENCE_OVERLAP):
            sequence_frames = frame_paths[i:i+SEQUENCE_SIZE]
            sequence_timestamps = timestamps[i:i+SEQUENCE_SIZE]
            
            if len(sequence_frames) == SEQUENCE_SIZE:  # Only use complete sequences
                sequences.append({
                    'frames': sequence_frames,
                    'timestamps': sequence_timestamps,
                    'start_time': sequence_timestamps[0],
                    'end_time': sequence_timestamps[-1]
                })
        
        return sequences

    def _create_sequence_analysis_prompt(self):
        """Create a prompt for analyzing sequences of frames"""
        return """You are an expert football analyst identifying key events from sequences of consecutive frames.

I'll show you a sequence of consecutive frames from a football match. Each frame is timestamped.

Focus ONLY on identifying these five key football events:

1. GOAL
   - Definition: Ball completely crosses goal line between posts and under crossbar
   - Visual indicators:
     * Shot being taken, ball crossing line/entering net
     * Player celebrations with raised arms
     * Goalkeeper retrieving ball from net
     * Referee pointing to center circle

2. PENALTY
   - Definition: Direct free kick from penalty spot after foul in penalty area
   - Visual indicators:
     * Referee pointing to penalty spot
     * Player positioning on/around penalty spot
     * Goalkeeper on goal line
     * Penalty being taken

3. CORNER KICK
   - Definition: Ball placed at corner arc after crossing goal line off defender
   - Visual indicators:
     * Player at corner flag
     * Players positioning in penalty area
     * Ball being delivered from corner position

4. FREE KICK
   - Definition: Set piece after a foul
   - Visual indicators:
     * Defensive wall formation
     * Player(s) standing over the ball
     * Referee positioning defenders at proper distance

5. CARD (YELLOW/RED)
   - Definition: Disciplinary action by referee
   - Visual indicators:
     * Referee showing card to specific player
     * Player reaction to card

IMPORTANT GUIDELINES:
- ONLY identify these 5 specific events - no other football actions
- Look at the chronological sequence to understand the flow of play
- Provide the EXACT time range for the event based on the frame timestamps
- Be very specific about what visual evidence you see
- If you're uncertain about an event, DO NOT include it

Return your findings in this EXACT format:
[MM:SS-MM:SS] EVENT_TYPE: Brief description with specific visual evidence

Example:
[35:20-35:28] GOAL: Ball struck from outside box, enters net past goalkeeper's dive, followed by player celebration with raised arms

If no clear events in these frames, respond with "No key events detected in this sequence."
"""

    def _analyze_frame_sequences(self, sequences, segment_idx, total_segments):
        """Analyze frame sequences using OpenAI API"""
        if not sequences:
            return []
        
        segment_events = []
        
        # Create progress bar for sequences
        with tqdm(total=len(sequences), desc="Sequence Analysis") as seq_pbar:
            for seq_idx, sequence in enumerate(sequences):
                frames = sequence['frames']
                timestamps = sequence['timestamps']
                start_time = sequence['start_time']
                end_time = sequence['end_time']
                
                start_time_str = self._format_time(start_time)
                end_time_str = self._format_time(end_time)
                
                # Create message content
                message_content = [
                    {"type": "text", "text": f"""
                    Analyze these {SEQUENCE_SIZE} consecutive frames from a football match.
                    
                    Time range: {start_time_str} to {end_time_str}
                    
                    The frames are in chronological order.
                    Each frame has its timestamp displayed.
                    
                    Identify if any of these SPECIFIC football events are occurring:
                    - GOAL
                    - PENALTY
                    - CORNER KICK
                    - FREE KICK
                    - CARD (YELLOW/RED)
                    
                    ONLY report events you're absolutely certain about with precise visual evidence.
                    """}
                ]
                
                # Add frames to message
                for frame_path in frames:
                    try:
                        with open(frame_path, "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                            message_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            })
                    except Exception as e:
                        print(f"Error encoding image {frame_path}: {str(e)}")
                
                # Define frame_ids for reference in results
                frame_ids = [os.path.basename(frame) for frame in frames]
                
                # API call
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": self._create_sequence_analysis_prompt()},
                            {"role": "user", "content": message_content}
                        ],
                        temperature=0.2
                    )
                    
                    event_text = response.choices[0].message.content
                    
                    # Parse events with time ranges
                    events = self._parse_events(event_text, start_time, end_time, frame_ids)
                    segment_events.extend(events)
                    
                    # Print events found in this sequence
                    if events:
                        print(f"  Sequence {seq_idx+1}/{len(sequences)}: Found {len(events)} events")
                        for event in events:
                            print(f"    [{self._format_time(event['start_time'])}-{self._format_time(event['end_time'])}] {event['event_type']}")
                            print(f"    Frames: {', '.join(event['frame_ids'])}")
                    
                except Exception as e:
                    print(f"Error analyzing sequence {start_time_str}-{end_time_str}: {str(e)}")
                    time.sleep(2)  # Wait before retry
                
                seq_pbar.update(1)
        
        # Post-process: merge similar events that are close in time
        merged_events = self._merge_events(segment_events)
        
        print(f"Segment {segment_idx+1}/{total_segments} complete: Found {len(merged_events)} distinct events")
        
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
                    
                    # Only accept the 5 specific event types we're looking for
                    if event_type in ["GOAL", "PENALTY", "CORNER KICK", "FREE KICK", "CARD"]:
                        events.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'event_type': event_type,
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
                'start_time': event['start_time'],  # Keep this for sorting
                'event_type': event['event_type']
            })
        
        # Sort by start time
        formatted_events.sort(key=lambda x: x['start_time'])
        
        return formatted_events

    def analyze(self):
        """Main analysis pipeline"""
        try:
            # Analyze the video
            events = self.analyze_video()
            
            # Clean up temporary files but keep frames
            self._cleanup(keep_frames=True)
            
            return events
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            self._cleanup(keep_frames=True)
            return []
    
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
def analyze_football_video(video_path, output_path=None):
    """Analyze football video and save results"""
    if not output_path:
        output_path = os.path.splitext(video_path)[0] + "_events.txt"
    
    analyzer = FootballAnalyzer(video_path)
    events = analyzer.analyze()
    
    # Save results
    with open(output_path, 'w') as f:
        f.write("FOOTBALL MATCH KEY EVENTS\n")
        f.write("==========================\n\n")
        for event in events:
            f.write(f"{event['event_string']}\n")
            f.write(f"  Frames: {', '.join(event['frame_ids'])}\n\n")
    
    print(f"\nResults saved to {output_path}")
    print("\nDetected Events:")
    for event in events:
        print(event['event_string'])
        print(f"  Frames: {', '.join(event['frame_ids'])}")
        print()
    
    return events

# Streamlit app function
def streamlit_app():
    st.title("Football Match Key Event Detection")
    st.write("Detects goals, penalties, corners, free kicks, and cards with precise time ranges")
    
    uploaded_file = st.file_uploader("Upload a football match video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        col1, col2 = st.columns(2)
        with col1:
            analyze_button = st.button("Start Analysis", type="primary")
        
        if analyze_button:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Starting analysis...")
            progress_bar.progress(10)
            
            try:
                status_text.text("Analyzing video...")
                progress_bar.progress(30)
                
                events = analyze_football_video("temp_video.mp4", "match_events.txt")
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                st.success(f"Detected {len(events)} key events")
                
                st.subheader("Key Football Events")
                for i, event in enumerate(events):
                    with st.expander(f"{event['event_string']}"):
                        st.write(event['event_string'])
                        
                        # Display frames for this event (up to 6)
                        frames_to_show = event['frame_ids'][:6]  # Limit to 6 frames
                        cols = st.columns(min(3, len(frames_to_show)))  # Up to 3 columns
                        
                        for j, frame_id in enumerate(frames_to_show):
                            frame_path = os.path.join(TEMP_DIR, "frames", frame_id)
                            if os.path.exists(frame_path):
                                cols[j % 3].image(frame_path, caption=f"Frame {frame_id}")
                
                with open("match_events.txt", "r") as f:
                    st.download_button(
                        label="Download Results",
                        data=f.read(),
                        file_name="football_events.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
            
            # Clean up
            if os.path.exists("temp_video.mp4"):
                os.remove("temp_video.mp4")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Football Match Event Analysis")
    parser.add_argument("--video", type=str, help="Path to football match video")
    parser.add_argument("--output", type=str, help="Path to output file (optional)")
    
    args = parser.parse_args()
    
    if args.video:
        analyze_football_video(args.video, args.output)
    else:
        # If no command line args, try to run as Streamlit app
        try:
            streamlit_app()
        except:
            print("To run as CLI: python script.py --video path/to/video.mp4")
            print("To run as Streamlit app: streamlit run script.py")