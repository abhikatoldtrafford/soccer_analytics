#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
enhanced_soccer_tracker.py
Production-ready soccer player tracking system using:
1) YOLO detection + Tracker (e.g., BoT-SORT/ByteTrack) with persist=True
2) TorchReID for robust Re-ID (e.g., osnet_ain_x1_0)
3) PaddleOCR for jersey number detection (optional, highlights work without it)
4) Feature clustering for better identity representation
5) Temporal consistency with buffer for improved matching
6) Formalized score fusion of appearance and jersey information
7) Spatial consistency (prevents merging players in the same frame)
8) Motion prediction for improved tracking during occlusions
9) Memory-efficient batch processing
10) Performance metrics and logging
11) Enhanced visualization options
12) Real-time streaming support
13) Per-player highlight package data generation for all players
"""

import os
import sys
import time
import cv2
import torch
import numpy as np
import pandas as pd
import logging
import traceback
import json
from datetime import datetime
from collections import defaultdict, Counter, deque
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Try importing matplotlib for better trajectory colors, but handle absence
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    logging.warning("Matplotlib not found. Trajectory visualization will use fallback colors.")

# Try importing sklearn for feature clustering
try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    logging.warning("scikit-learn not found. Feature clustering will be disabled.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("soccer_tracker.log")
    ]
)
logger = logging.getLogger("SoccerTracker")


# --------------------- ReID --------------------- #
class PlayerReIDTracker:
    """
    Maintains a set of global identities using TorchReID features.
    Enforces that no two track IDs in the same frame can merge into one global ID.
    Jersey numbers provide high-confidence identification override when available.

    Features:
    - Feature clustering for multiple appearance representations per player
    - Memory-efficient feature storage with max history limit
    - Cosine similarity-based matching
    - Jersey number voting system with high-confidence override
    - Spatial consistency enforcement
    - Formalized score fusion between appearance and jersey information
    """
    def __init__(self, model_name, model_weights, device='cuda', reid_threshold=0.9,
                 max_feat_history=100, jersey_boost=0.3, 
                 use_clustering=True, num_clusters=3, 
                 reid_weight=0.7, jersey_weight=0.3):
        """
        Initialize the ReID tracker.

        Args:
            model_name (str): TorchReID model name
            model_weights (str): Path to model weights
            device (str): 'cuda' or 'cpu'
            reid_threshold (float): Similarity threshold for matching
            max_feat_history (int): Maximum number of features to store per ID
            jersey_boost (float): Similarity boost when jersey numbers match
            use_clustering (bool): Whether to use feature clustering
            num_clusters (int): Number of clusters to maintain per identity
            reid_weight (float): Weight for ReID score in fusion (0-1)
            jersey_weight (float): Weight for jersey match score in fusion (0-1)
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.reid_threshold = reid_threshold
        self.max_feat_history = max_feat_history
        self.jersey_boost = jersey_boost
        
        # New clustering parameters
        self.use_clustering = use_clustering and CLUSTERING_AVAILABLE
        self.num_clusters = num_clusters
        
        # New fusion weights
        self.reid_weight = reid_weight
        self.jersey_weight = jersey_weight
        
        # Ensure weights sum to 1
        total_weight = self.reid_weight + self.jersey_weight
        if total_weight != 1.0:
            self.reid_weight /= total_weight
            self.jersey_weight /= total_weight

        # Load TorchReID Feature Extractor
        try:
            # First, ensure torchreid is imported
            import torchreid
            from torchreid.utils import FeatureExtractor

            logger.info(f"Initializing TorchReID with model: {model_name}")
            self.feature_extractor = FeatureExtractor(
                model_name=model_name,
                model_path=model_weights,
                device=self.device
            )
            logger.info("TorchReID initialized successfully")
        except ImportError:
            logger.error("Failed to import torchreid. Make sure it's installed.")
            raise
        except Exception as e:
            logger.error(f"Error initializing TorchReID: {str(e)}\n{traceback.format_exc()}")
            raise

        # Next assigned global ID
        self.next_global_id = 1

        # For storing appearance history (ReID features) for each global ID
        # Using deque for memory efficiency
        self.global_id_features = defaultdict(lambda: deque(maxlen=self.max_feat_history))
        
        # New: Store cluster centroids for each global ID when clustering is enabled
        self.global_id_clusters = {}  # {global_id: {'centroids': [...], 'counts': [...]}}

        # For storing jersey votes (jersey_num -> count)
        self.global_id_jersey_votes = defaultdict(Counter)

        # For mapping jersey numbers to global IDs
        # jersey_number -> {global_id: confidence}
        self.jersey_to_global_ids = defaultdict(dict)

        # Track when each jersey was first identified for a global ID
        self.jersey_first_seen = {}  # (global_id, jersey_num) -> frame_idx
        
        # New: Buffer for temporal matching (recent unmatched features)
        self.unmatched_buffer = []  # List of (feat, frame_idx, bbox) tuples

        # Performance metrics
        self.metrics = {
            'total_matches': 0,
            'jersey_matches': 0,
            'new_ids_created': 0,
            'avg_similarity': [],
            'jersey_overrides': 0,
            'cluster_updates': 0,
            'temporal_matches': 0
        }

    def _update_clusters(self, global_id):
        """
        Update feature clusters for a global ID.
        
        Args:
            global_id: The global ID to update clusters for
        """
        if not self.use_clustering or global_id not in self.global_id_features:
            return
            
        features = list(self.global_id_features[global_id])
        if len(features) < self.num_clusters * 2:  # Need sufficient samples for clustering
            return
            
        try:
            # Use MiniBatchKMeans for better performance with larger datasets
            features_array = np.vstack(features)
            n_clusters = min(self.num_clusters, len(features) // 2)  # Ensure we have at least 2 samples per cluster
            
            # Standardize features for better clustering
            feat_mean = np.mean(features_array, axis=0)
            feat_std = np.std(features_array, axis=0) + 1e-8  # Avoid division by zero
            features_standardized = (features_array - feat_mean) / feat_std
            
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=min(100, len(features)))
            labels = kmeans.fit_predict(features_standardized)
            
            # Get centroids and transform back to original feature space
            centroids_std = kmeans.cluster_centers_
            centroids = centroids_std * feat_std + feat_mean
            
            # Count samples per cluster for weighting
            counts = np.bincount(labels, minlength=n_clusters)
            
            # Store centroids and counts
            self.global_id_clusters[global_id] = {
                'centroids': centroids.tolist(),
                'counts': counts.tolist()
            }
            
            self.metrics['cluster_updates'] += 1
            logger.debug(f"Updated clusters for GID {global_id}: {n_clusters} clusters with counts {counts}")
            
        except Exception as e:
            logger.error(f"Error updating clusters for GID {global_id}: {str(e)}\n{traceback.format_exc()}")

    def extract_features(self, crops, batch_size=32):
        """
        Extract TorchReID features for a list of bounding-box crops.
        Processes in batches for memory efficiency.

        Args:
            crops (list): List of image crops
            batch_size (int): Batch size for processing

        Returns:
            list: CPU numpy feature vectors
        """
        if not crops:
            return []

        all_feats = []
        feat_dim = 512 # Default, will try to determine from model name

        try:
            # Try to determine expected feature dim from model name
            if hasattr(self.feature_extractor, 'model_name'):
                 if "resnet" in self.feature_extractor.model_name: feat_dim = 2048
                 # Add other model families if needed
        except AttributeError: pass # Ignore if model_name not available

        # Process in batches to save memory
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i+batch_size]
            if not batch: continue # Skip empty batch

            try:
                with torch.no_grad(): # Ensure no gradients are computed
                    feats = self.feature_extractor(batch)
                # Convert each feature to CPU numpy
                batch_feats = [f.cpu().numpy() if torch.is_tensor(f) else f for f in feats]
                all_feats.extend(batch_feats)
            except Exception as e:
                # Log the error and append zero vectors
                logger.error(f"Error extracting features for batch starting at index {i}: {str(e)}\n{traceback.format_exc()}")
                # Fill with zeros for failed crops in this batch
                for _ in range(len(batch)):
                    all_feats.append(np.zeros(feat_dim))
        return all_feats

    def compute_similarity(self, featA, featB):
        """
        Cosine similarity between two vectors. Range [-1..1].

        Args:
            featA (numpy.ndarray): First feature vector
            featB (numpy.ndarray): Second feature vector

        Returns:
            float: Similarity score
        """
        # Handle zero or invalid vectors
        if featA is None or featB is None or np.all(featA == 0) or np.all(featB == 0):
            return 0.0
        if featA.shape != featB.shape:
             logger.warning(f"Feature shape mismatch: {featA.shape} vs {featB.shape}")
             return 0.0

        # Use float32 for stability in norm calculation
        featA = featA.astype(np.float32)
        featB = featB.astype(np.float32)

        normA = np.linalg.norm(featA)
        normB = np.linalg.norm(featB)

        if normA < 1e-6 or normB < 1e-6: # Check for very small norms
            return 0.0

        dot = np.dot(featA, featB)
        denom = (normA * normB)
        sim = dot / denom
        return np.clip(sim, -1.0, 1.0) # Clip to ensure valid range

    def compute_clustered_similarity(self, feat, global_id):
        """
        Compute similarity using clustered representations.
        
        Args:
            feat (numpy.ndarray): Feature vector to match
            global_id (int): Global ID to match against
            
        Returns:
            float: Best similarity score
        """
        # Use standard approach if clustering not enabled or no clusters yet
        if not self.use_clustering or global_id not in self.global_id_clusters:
            if global_id not in self.global_id_features or not self.global_id_features[global_id]:
                return 0.0
                
            # Use average feature approach
            feats_array = np.stack(list(self.global_id_features[global_id]))
            avg_feat = np.mean(feats_array, axis=0)
            return self.compute_similarity(feat, avg_feat)
        
        # Get cluster info
        cluster_info = self.global_id_clusters[global_id]
        centroids = np.array(cluster_info['centroids'])
        counts = np.array(cluster_info['counts'])
        
        # Compute similarity to each centroid
        similarities = np.array([self.compute_similarity(feat, centroid) for centroid in centroids])
        
        # Weight by cluster size if valid (normalize counts first)
        if np.sum(counts) > 0:
            weights = counts / np.sum(counts)
            weighted_sim = np.sum(similarities * weights)
            # Also consider max similarity for strong matches
            max_sim = np.max(similarities)
            # Use a balanced approach - 70% max sim, 30% weighted average
            balanced_sim = 0.7 * max_sim + 0.3 * weighted_sim
            return balanced_sim
        else:
            # Fallback to max similarity if counts are problematic
            return np.max(similarities)

    def compute_jersey_score(self, jersey_num, jersey_conf, global_id):
        """
        Compute jersey match score between a detected jersey and a global ID.
        
        Args:
            jersey_num (int): Detected jersey number
            jersey_conf (float): Confidence in jersey detection
            global_id (int): Global ID to check against
            
        Returns:
            float: Jersey match score [0..1]
        """
        if jersey_num is None or jersey_conf < 0.3:
            return 0.0
            
        # Get best jersey for this global ID
        best_jersey, best_conf = self.get_best_jersey(global_id)
        
        if best_jersey is None:
            # No jersey for this global ID yet
            return 0.1  # Small positive score for the potential
        
        if best_jersey == jersey_num:
            # Same jersey - score based on both confidences
            match_score = jersey_conf * best_conf * 1.5  # Boost matching jerseys
            return min(1.0, match_score)  # Cap at 1.0
        else:
            # Different jersey - penalty
            # Higher penalty for more confident detections
            mismatch_penalty = -0.5 * jersey_conf * best_conf
            return max(0.0, mismatch_penalty)  # Floor at 0.0

    def compute_fused_score(self, reid_similarity, jersey_num, jersey_conf, global_id):
        """
        Computes a fused matching score combining ReID and jersey information.
        
        Args:
            reid_similarity (float): ReID similarity score [-1..1]
            jersey_num (int): Detected jersey number
            jersey_conf (float): Confidence in jersey detection
            global_id (int): Global ID to check against
            
        Returns:
            float: Fused score for matching
        """
        # Normalize ReID similarity from [-1..1] to [0..1]
        reid_score = (reid_similarity + 1) / 2.0
        
        # Get jersey match score
        jersey_score = self.compute_jersey_score(jersey_num, jersey_conf, global_id)
        
        # Fuse scores using configured weights
        fused_score = (self.reid_weight * reid_score) + (self.jersey_weight * jersey_score)
        
        # Debug log for significant matches
        if fused_score > 0.7:
            logger.debug(f"Fused score for GID {global_id}: {fused_score:.3f} " 
                        f"(ReID: {reid_score:.3f}*{self.reid_weight:.2f}, "
                        f"Jersey: {jersey_score:.3f}*{self.jersey_weight:.2f})")
            
        return fused_score

    def add_to_temporal_buffer(self, feat, frame_idx, bbox):
        """
        Add an unmatched feature to the temporal buffer.
        
        Args:
            feat (numpy.ndarray): Feature vector
            frame_idx (int): Frame index
            bbox (list): Bounding box [x1, y1, x2, y2]
        """
        self.unmatched_buffer.append((feat, frame_idx, bbox))
        
        # Keep buffer size reasonable (last 5 frames worth, assuming ~10 unmatched per frame)
        max_buffer_size = 50
        if len(self.unmatched_buffer) > max_buffer_size:
            self.unmatched_buffer = self.unmatched_buffer[-max_buffer_size:]

    def match_to_global_id(self, new_feat, forbidden_gids, jersey_num=None, jersey_conf=0.0, frame_idx=0, bbox=None):
        """
        Attempt to match new_feat to an existing global ID, skipping any in 'forbidden_gids'.
        If no match above reid_threshold, create new ID.
        If jersey_num is provided with high confidence, it gets priority for matching.

        Args:
            new_feat (numpy.ndarray): The new appearance feature (CPU numpy)
            forbidden_gids (set): Set of global IDs active in the same frame (to skip)
            jersey_num (int, optional): Detected jersey number
            jersey_conf (float, optional): Confidence in jersey number detection
            frame_idx (int): Current frame index
            bbox (list, optional): Bounding box [x1, y1, x2, y2]

        Returns:
            int: The matched or newly created global_id
        """
        best_gid = None
        best_score = 0.5  # Minimum score threshold
        
        # --- 1. Direct high-confidence jersey matching ---
        if jersey_num is not None and jersey_conf >= 0.9:
            if jersey_num in self.jersey_to_global_ids:
                # Find the GID most strongly associated with this jersey
                best_jersey_gid = -1
                max_assoc_conf = 0.0
                for gid, assoc_conf in self.jersey_to_global_ids[jersey_num].items():
                    if gid not in forbidden_gids and assoc_conf > max_assoc_conf:
                        max_assoc_conf = assoc_conf
                        best_jersey_gid = gid

                if best_jersey_gid != -1 and max_assoc_conf > 0.7: # Check if association is reasonably strong
                    self.metrics['jersey_matches'] += 1
                    best_gid = best_jersey_gid
                    logger.debug(f"Frame {frame_idx}: High-conf jersey #{jersey_num} direct match to GID {best_gid}")
        
        # --- 2. Try temporal matching from buffer (if no direct jersey match) ---
        if best_gid is None and bbox is not None:
            # Create a temporal matching score considering:
            # - Feature similarity
            # - Time difference (fresher is better)
            # - Jersey number match
            # - IoU if bboxes are close
            
            # Convert bbox to center point and size
            x1, y1, x2, y2 = bbox
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            bbox_w, bbox_h = x2 - x1, y2 - y1
            
            # Calculate temporal scores for all buffer entries not in forbidden_gids
            temporal_scores = {}
            
            for buffered_feat, buffered_frame, buffered_bbox in self.unmatched_buffer:
                # Skip if too old (> 5 frames)
                if frame_idx - buffered_frame > 5:
                    continue
                    
                # Compute feature similarity
                sim = self.compute_similarity(new_feat, buffered_feat)
                
                # Skip if similarity is too low
                if sim < 0.7:
                    continue
                
                # Calculate spatial proximity if bboxes available
                spatial_score = 0.0
                if buffered_bbox is not None:
                    b_x1, b_y1, b_x2, b_y2 = buffered_bbox
                    b_center_x, b_center_y = (b_x1 + b_x2) / 2, (b_y1 + b_y2) / 2
                    
                    # Distance between centers (normalized by average size)
                    avg_size = (bbox_w + bbox_h + (b_x2 - b_x1) + (b_y2 - b_y1)) / 4
                    dist = np.sqrt((center_x - b_center_x)**2 + (center_y - b_center_y)**2)
                    dist_score = max(0, 1 - dist / (avg_size * 2))
                    
                    # Size similarity
                    b_w, b_h = b_x2 - b_x1, b_y2 - b_y1
                    size_ratio = min(bbox_w * bbox_h, b_w * b_h) / max(bbox_w * bbox_h, b_w * b_h)
                    
                    spatial_score = 0.7 * dist_score + 0.3 * size_ratio
                
                # Temporal score (fresher is better)
                time_score = max(0, 1 - (frame_idx - buffered_frame) / 5)
                
                # Combined score
                combined_score = 0.6 * sim + 0.3 * spatial_score + 0.1 * time_score
                
                # Track best match
                if combined_score > best_score and combined_score > 0.75:
                    temporal_scores[combined_score] = (buffered_feat, buffered_frame)
            
            # If we found any good temporal matches
            if temporal_scores:
                best_temp_score = max(temporal_scores.keys())
                best_temp_feat, best_temp_frame = temporal_scores[best_temp_score]
                
                # Use the match for this new object
                # We'll check clustered similarity against all eligible GIDs
                for gid in self.global_id_features:
                    if gid in forbidden_gids:
                        continue
                        
                    # Check similarity with this GID using features
                    reid_sim = self.compute_clustered_similarity(best_temp_feat, gid)
                    
                    # Convert to fused score
                    fused_score = self.compute_fused_score(reid_sim, jersey_num, jersey_conf, gid)
                    
                    # If good match, use this GID
                    if fused_score > best_score:
                        best_score = fused_score
                        best_gid = gid
                        
                if best_gid is not None:
                    logger.debug(f"Frame {frame_idx}: Temporal buffer match to GID {best_gid} (score: {best_score:.3f})")
                    self.metrics['temporal_matches'] += 1
        
        # --- 3. Standard feature matching if no direct match yet ---
        if best_gid is None:
            for gid in self.global_id_features:
                if gid in forbidden_gids:
                    continue
                
                # Compute ReID similarity using clusters if available
                reid_sim = self.compute_clustered_similarity(new_feat, gid)
                
                # Convert to fused score
                fused_score = self.compute_fused_score(reid_sim, jersey_num, jersey_conf, gid)
                
                if fused_score > best_score:
                    best_score = fused_score
                    best_gid = gid
                    logger.debug(f"Frame {frame_idx}: Standard match to GID {best_gid} (score: {best_score:.3f})")

        # --- 4. Create new ID if no good match ---
        if best_gid is None or best_score < self.reid_threshold:
            # Create new global ID if no match above threshold
            best_gid = self.next_global_id
            self.next_global_id += 1
            self.metrics['new_ids_created'] += 1
            logger.debug(f"Frame {frame_idx}: Created new GID {best_gid}")
        else:
            # Record the successful match
            self.metrics['total_matches'] += 1
            # Only track meaningful similarity scores
            if best_score > 0.6:
                self.metrics['avg_similarity'].append(best_score)

        # --- 5. Update ID with new information ---
        # Add new feature to history
        self.global_id_features[best_gid].append(new_feat)
        
        # Update jersey votes if applicable
        if jersey_num is not None and jersey_conf > 0.5:
            self.global_id_jersey_votes[best_gid][jersey_num] += 1
            self._update_jersey_mapping(best_gid)

            # Track first time this jersey was seen for this GID
            if (best_gid, jersey_num) not in self.jersey_first_seen:
                self.jersey_first_seen[(best_gid, jersey_num)] = frame_idx
                
        # Update clusters if enough features have accumulated
        if len(self.global_id_features[best_gid]) >= self.num_clusters * 2:
            # Only update clusters periodically or for new IDs
            update_freq = 10  # Update every 10 frames for existing IDs
            if (best_gid not in self.global_id_clusters) or (frame_idx % update_freq == 0):
                self._update_clusters(best_gid)

        return best_gid

    def _update_jersey_mapping(self, global_id):
        """
        Update the jersey number to global ID mapping based on latest votes.

        Args:
            global_id (int): Global ID to update mapping for
        """
        jersey_num, confidence = self.get_best_jersey(global_id)
        if jersey_num is not None:
            # Update mapping for this jersey number
            self.jersey_to_global_ids[jersey_num][global_id] = confidence
            # Clean up old mappings if this GID previously had a different jersey
            for j_num, gid_map in list(self.jersey_to_global_ids.items()):
                 if global_id in gid_map and j_num != jersey_num:
                     del self.jersey_to_global_ids[j_num][global_id]
                     if not self.jersey_to_global_ids[j_num]: # Remove empty jersey entry
                         del self.jersey_to_global_ids[j_num]

    def get_best_jersey(self, global_id):
        """
        Get the most voted jersey number for a global ID.

        Args:
            global_id (int): Global ID

        Returns:
            tuple: (jersey_number, confidence) or (None, 0)
        """
        if global_id not in self.global_id_jersey_votes or not self.global_id_jersey_votes[global_id]:
            return None, 0

        # Get the most common jersey number
        jersey_counter = self.global_id_jersey_votes[global_id]
        total_votes = sum(jersey_counter.values())
        if total_votes == 0: return None, 0

        most_common = jersey_counter.most_common(1)[0]
        jersey_num, count = most_common

        # Calculate confidence as proportion of votes
        confidence = count / total_votes

        return jersey_num, confidence

    def override_with_jersey(self, global_id, jersey_num, jersey_conf, frame_idx=0):
        """
        Override a global ID's jersey number with a high-confidence detection.
        
        Args:
            global_id (int): Global ID to override
            jersey_num (int): New jersey number
            jersey_conf (float): Confidence in jersey number detection
            frame_idx (int): Current frame index
            
        Returns:
            bool: True if override was performed
        """
        if jersey_conf < 0.85:  # Keep high threshold for overrides
            return False
            
        # Check if this is already the best jersey
        current_jersey, current_conf = self.get_best_jersey(global_id)
        
        # Track first time this jersey was seen for this GID
        if (global_id, jersey_num) not in self.jersey_first_seen:
            self.jersey_first_seen[(global_id, jersey_num)] = frame_idx
        
        # If already matching, just return
        if current_jersey == jersey_num:
            return False
        
        # If current jersey is stable and high confidence,
        # require even higher confidence to switch
        if current_jersey is not None and current_conf >= 0.8:
            # Check if current jersey has been stable for some time
            current_key = (global_id, current_jersey)
            if current_key in self.jersey_first_seen:
                frames_with_current = frame_idx - self.jersey_first_seen[current_key]
                if frames_with_current > 30:  # If stable for >30 frames
                    # Require very high confidence to override established jersey
                    if jersey_conf < 0.95:
                        return False
        
        # Check if this jersey is strongly associated with another global ID
        if jersey_num in self.jersey_to_global_ids:
            for other_gid, other_conf in self.jersey_to_global_ids[jersey_num].items():
                if other_gid != global_id and other_conf >= 0.85:
                    # This jersey is strongly associated with another player
                    
                    # Don't merge if both GIDs have stable but different jerseys
                    other_jersey, _ = self.get_best_jersey(other_gid)
                    if current_jersey is not None and other_jersey is not None and current_jersey != other_jersey:
                        # Both players have different stable jerseys, don't merge
                        if (global_id, current_jersey) in self.jersey_first_seen and \
                        (other_gid, other_jersey) in self.jersey_first_seen:
                            frames_with_current = frame_idx - self.jersey_first_seen[(global_id, current_jersey)]
                            frames_with_other = frame_idx - self.jersey_first_seen[(other_gid, other_jersey)]
                            
                            # If both jerseys are stable for some time, don't merge
                            if frames_with_current > 20 and frames_with_other > 20:
                                logger.info(f"Preventing merge of GID {global_id} (jersey #{current_jersey}) into GID {other_gid} (jersey #{other_jersey}) despite new detection of #{jersey_num}")
                                return False
                    
                    # Proceed with merge if no stability issues
                    logger.info(f"Merging global ID {global_id} into {other_gid} based on jersey number {jersey_num}")
                    self._merge_global_ids(global_id, other_gid)
                    self.metrics['jersey_overrides'] += 1
                    return True
        
        # Add a strong vote for this jersey number
        boost_votes = max(5, len(self.global_id_jersey_votes[global_id])) 
        self.global_id_jersey_votes[global_id][jersey_num] += boost_votes
        self._update_jersey_mapping(global_id)
        self.metrics['jersey_overrides'] += 1
        return True

    def _merge_global_ids(self, source_gid, target_gid):
        """
        Merge source global ID into target global ID. Updates features, votes, and first_seen.

        Args:
            source_gid (int): Source global ID to merge from
            target_gid (int): Target global ID to merge into
        """
        if source_gid == target_gid: return # Cannot merge into self

        logger.info(f"Executing merge: {source_gid} -> {target_gid}")

        # Merge features (ensure target deque doesn't overflow if combined history is large)
        source_feats = list(self.global_id_features.get(source_gid, [])) # Use .get for safety
        target_feats_deque = self.global_id_features[target_gid]
        # Add source features, respecting maxlen of the target deque
        for feat in reversed(source_feats): # Add older source features first
             if len(target_feats_deque) < target_feats_deque.maxlen:
                 target_feats_deque.appendleft(feat) # Add to the beginning if space
             else:
                 break # Stop if target deque is full

        # Merge jersey votes
        source_votes = self.global_id_jersey_votes.get(source_gid, Counter())
        for jersey_num, count in source_votes.items():
            self.global_id_jersey_votes[target_gid][jersey_num] += count

            # Merge first seen frame info - keep the *earliest* frame
            source_key = (source_gid, jersey_num)
            target_key = (target_gid, jersey_num)

            source_frame = self.jersey_first_seen.get(source_key)
            target_frame = self.jersey_first_seen.get(target_key)

            if source_frame is not None:
                if target_frame is not None:
                    self.jersey_first_seen[target_key] = min(source_frame, target_frame)
                else:
                    self.jersey_first_seen[target_key] = source_frame
                # Remove source entry after processing
                if source_key in self.jersey_first_seen:
                    del self.jersey_first_seen[source_key]

        # Merge clusters if both exist
        if source_gid in self.global_id_clusters and target_gid in self.global_id_clusters:
            # For simplicity, we'll just trigger a recomputation of clusters on the target
            # based on the newly merged features
            self._update_clusters(target_gid)
        elif source_gid in self.global_id_clusters:
            # Copy source clusters to target if target has none
            self.global_id_clusters[target_gid] = self.global_id_clusters[source_gid]

        # Clear source ID data
        self.global_id_features.pop(source_gid, None)
        self.global_id_jersey_votes.pop(source_gid, None)
        self.global_id_clusters.pop(source_gid, None)

        # Clean up jersey_to_global_ids mapping for the source_gid
        for j_num, gid_map in list(self.jersey_to_global_ids.items()):
            if source_gid in gid_map:
                del self.jersey_to_global_ids[j_num][source_gid]
                if not self.jersey_to_global_ids[j_num]:
                    del self.jersey_to_global_ids[j_num]

        # Update mappings for the target ID based on merged votes
        self._update_jersey_mapping(target_gid)

        # Note: The calling function (SoccerTracker) needs to update trackid_to_globalid map

    def get_metrics(self):
        """
        Get performance metrics for the ReID system.

        Returns:
            dict: Performance metrics
        """
        avg_sim = np.mean(self.metrics['avg_similarity']) if self.metrics['avg_similarity'] else 0
        metrics = {
            'total_matches': self.metrics['total_matches'],
            'new_ids_created': self.metrics['new_ids_created'],
            'avg_similarity': float(avg_sim), # Ensure float type
            'total_global_ids_ever': self.next_global_id - 1,
            'active_global_ids': len(self.global_id_features), # IDs currently holding features
            'ids_with_jerseys': sum(1 for counter in self.global_id_jersey_votes.values() if counter),
            'jersey_overrides_merges': self.metrics['jersey_overrides'],
            'cluster_updates': self.metrics['cluster_updates'],
            'temporal_matches': self.metrics['temporal_matches']
        }
        return metrics


# --------------------- OCR --------------------- #
class JerseyOCR:
    """
    PaddleOCR-based numeric detection for jersey numbers.
    Features:
    - Confidence thresholding
    - Error handling
    - GPU/CPU support
    - Whole-frame OCR with IoU matching
    """
    # Adjusted defaults based on typical performance
    def __init__(self, conf_threshold=0.5, high_conf_threshold=0.8, use_gpu=True):
        """
        Initialize the OCR detector.

        Args:
            conf_threshold (float): Confidence threshold for detection
            high_conf_threshold (float): High confidence threshold for strong jersey evidence
            use_gpu (bool): Whether to use GPU
        """
        try:
            logger.info("Initializing PaddleOCR for jersey detection")
            # Reduce log verbosity of PaddleOCR itself
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=(use_gpu and torch.cuda.is_available()),
                show_log=False # Suppress PaddleOCR internal logs
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing PaddleOCR: {str(e)}\n{traceback.format_exc()}")
            raise

        self.conf_threshold = conf_threshold
        self.high_conf_threshold = high_conf_threshold

        # Performance metrics
        self.metrics = {
            'total_ocr_runs': 0,
            'successful_detections': 0, # Number of jerseys found meeting conf_threshold
            'high_conf_detections': 0 # Number meeting high_conf_threshold
        }

    def detect_whole_frame(self, frame):
        """
        Run OCR on the entire frame and return all jersey number detections.
        
        Args:
            frame (numpy.ndarray): Full video frame
            
        Returns:
            list: List of dictionaries containing jersey number detections
                with format {'number': int, 'confidence': float, 'bbox': [x1,y1,x2,y2]}
        """
        self.metrics['total_ocr_runs'] += 1
        try:
            result = self.ocr.ocr(frame, det=True, cls=True)
            if not result or not result[0]:
                return []
                
            jersey_detections = []
            
            for line in result[0]:
                bbox, (txt, cf) = line
                txt = txt.strip()
                digits_only = "".join(filter(str.isdigit, txt))
                if cf > self.conf_threshold and digits_only:
                    val = int(digits_only)
                    if 1 <= val <= 99:  # Valid jersey numbers
                        # Convert OCR bbox format to [x1,y1,x2,y2]
                        # OCR provides [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                        
                        # IMPROVEMENT 1: Add sanity checks on bbox size and aspect ratio
                        w_box, h_box = abs(x2 - x1), abs(y2 - y1)
                        
                        # Skip if box is too small
                        if w_box < 5 or h_box < 5 or w_box * h_box < 50:
                            continue
                            
                        # Check aspect ratio - jersey numbers are typically taller than wide
                        # or roughly square, but rarely very wide and short
                        aspect_ratio = w_box / max(h_box, 1)  # Avoid division by zero
                        if aspect_ratio > 2.5:  # Too wide for a typical jersey number
                            continue
                        
                        # IMPROVEMENT 2: For high confidence, require more balanced aspect ratio
                        if cf >= self.high_conf_threshold and aspect_ratio > 1.5:
                            cf = cf * 0.8  # Reduce confidence for unusually wide jersey numbers
                        
                        jersey_detections.append({
                            'number': val,
                            'confidence': cf,
                            'bbox': [x1, y1, x2, y2]
                        })
                
            return jersey_detections
            
        except Exception as e:
            logger.error(f"Error in whole frame OCR detection: {str(e)}")
            return []

    def calculate_iou(self, box1, box2):
        """
        Calculate IoU (Intersection over Union) between two bounding boxes.

        Args:
            box1 (list): First bounding box [x1, y1, x2, y2]
            box2 (list): Second bounding box [x1, y1, x2, y2]

        Returns:
            float: IoU value
        """
        x1_i = max(box1[0], box2[0])
        y1_i = max(box1[1], box2[1])
        x2_i = min(box1[2], box2[2])
        y2_i = min(box1[3], box2[3])

        # Calculate intersection area
        intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)

        if intersection == 0:
            return 0.0

        # Calculate area of each box
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate IoU
        union = box1_area + box2_area - intersection
        iou = intersection / union if union > 0 else 0.0
        return max(0.0, min(iou, 1.0)) # Ensure value is in [0,1]

    def get_metrics(self):
        """
        Get performance metrics for OCR.

        Returns:
            dict: Performance metrics
        """
        success_rate = (self.metrics['successful_detections'] / self.metrics['total_ocr_runs']
                        if self.metrics['total_ocr_runs'] > 0 else 0)

        return {
            'total_ocr_runs': self.metrics['total_ocr_runs'],
            'successful_jersey_detections': self.metrics['successful_detections'],
            'high_conf_jersey_detections': self.metrics['high_conf_detections'],
            'ocr_success_rate_per_run': success_rate
        }


# --------------------- Tracker --------------------- #
class SoccerTracker:
    """
    YOLO detection + MOT tracker => short-term track IDs.
    Merges track IDs into consistent ReID-based global IDs using appearance,
    jersey numbers (via whole-frame OCR + IoU), motion prediction, and temporal/spatial consistency.
    Generates data for per-player highlight packages for all tracked players.
    """
    # Updated defaults based on user's last code/request
    def __init__(self,
                 yolo_model_path: str,
                 reid_model_name: str = "osnet_ain_x1_0",
                 reid_weights_path: str = "osnet_ain_x1_0_market1501.pth",
                 conf: float = 0.3,
                 reid_threshold: float = 0.8, # User requested 0.8
                 tracker_cfg: str = "custom.yaml", # User provided custom.yaml
                 device: str = 'cuda',
                 ocr_interval: int = 5,
                 max_history: int = 900, # User increased
                 temporal_window: int = 300, # User increased
                 iou_match_threshold: float = 0.3, # Slightly increased from 0.2
                 fps: float = 30.0,
                 motion_prediction_weight: float = 0.3, # Weight for motion prediction in matching
                 use_clustering: bool = True,
                 reid_weight: float = 0.7, 
                 jersey_weight: float = 0.3):
        """
        Initialize the soccer tracker.
        
        Args:
            yolo_model_path (str): Path to YOLO model
            reid_model_name (str): TorchReID model name
            reid_weights_path (str): Path to ReID model weights
            conf (float): Confidence threshold for detection
            reid_threshold (float): Similarity threshold for ReID
            tracker_cfg (str): Tracker configuration
            device (str): 'cuda' or 'cpu'
            ocr_interval (int): Run OCR every N frames
            max_history (int): Maximum track history to keep
            temporal_window (int): Max frames to consider for temporal consistency
            iou_match_threshold (float): Minimum IoU for OCR-player box association
            fps (float): Video frames per second
            motion_prediction_weight (float): Weight for motion prediction in matching
            use_clustering (bool): Whether to use feature clustering
            reid_weight (float): Weight for ReID score in fusion formula
            jersey_weight (float): Weight for jersey score in fusion formula
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        self.conf = conf
        self.tracker_cfg = tracker_cfg
        self.ocr_interval = ocr_interval
        self.max_history = max_history
        self.temporal_window = temporal_window
        self.iou_match_threshold = iou_match_threshold
        self.fps = fps if fps > 0 else 30.0
        self.motion_prediction_weight = np.clip(motion_prediction_weight, 0.0, 1.0) # Ensure weight is valid

        # Performance tracking
        self.start_time = time.time()
        self.processed_frames = 0
        self.processing_times = []
        self._reappearances_matched = 0
        
        # New: Temporal buffer for unmatched tracks
        self.unmatched_temporal_buffer = []  # List of (frame_idx, tracks) tuples
        self.max_buffer_frames = 5  # Keep track of last 5 frames for temporal matching

        # Initialize YOLO
        try:
            logger.info(f"Loading YOLO model from {yolo_model_path}")
            self.yolo = YOLO(yolo_model_path)
            _ = self.yolo(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False) # Dummy inference
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}\n{traceback.format_exc()}")
            raise

        # Initialize ReID
        try:
            logger.info(f"Initializing TorchReID tracker with {reid_model_name}")
            self.reid = PlayerReIDTracker(
                model_name=reid_model_name, model_weights=reid_weights_path, device=self.device,
                reid_threshold=reid_threshold, max_feat_history=max_history,
                use_clustering=use_clustering, reid_weight=reid_weight, jersey_weight=jersey_weight
            )
            logger.info("ReID tracker initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ReID tracker: {str(e)}\n{traceback.format_exc()}")
            raise

        # Initialize OCR
        try:
            logger.info("Initializing Jersey OCR")
            self.ocr = JerseyOCR(use_gpu=(self.device == 'cuda'))
            logger.info("OCR initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OCR: {str(e)}\n{traceback.format_exc()}")
            raise

        # Tracking state
        self.trackid_to_globalid = {}
        self.frame_idx = 0
        self.track_history_viz = defaultdict(lambda: deque(maxlen=max(10, self.max_history // 5))) # Shorter viz history
        self.global_id_last_seen = {}
        self.inactive_tracks = {}
        self.global_id_velocity = {} # Store smoothed velocity (vx, vy)

        # Highlight data collection: key is jersey_num (int) or "gid_{gid}" (str)
        self.player_tracking_data = defaultdict(list)

        logger.info(f"SoccerTracker initialized successfully on {self.device}. FPS set to {self.fps:.2f}.")

    
    def _calculate_velocity(self, global_id, current_pos):
        """Calculates smoothed velocity using previous state."""
        vx, vy = 0.0, 0.0 # Default to zero velocity
        if global_id in self.global_id_last_seen:
            last_data = self.global_id_last_seen[global_id]
            last_pos = last_data.get('position')
            last_frame = last_data.get('frame')

            if last_pos is not None and last_frame is not None and self.frame_idx > last_frame:
                dt = self.frame_idx - last_frame
                # Instantaneous velocity
                inst_vx = (current_pos[0] - last_pos[0]) / dt
                inst_vy = (current_pos[1] - last_pos[1]) / dt

                # Apply exponential smoothing if previous velocity exists
                if global_id in self.global_id_velocity:
                    prev_vx, prev_vy = self.global_id_velocity[global_id]
                    # Adaptive smoothing factor (more weight to new data if dt is large or velocity changes drastically)
                    # A simple adaptive alpha:
                    alpha = np.clip(0.1 + 0.8 / dt, 0.1, 0.9)
                    vx = alpha * inst_vx + (1 - alpha) * prev_vx
                    vy = alpha * inst_vy + (1 - alpha) * prev_vy
                else:
                    vx, vy = inst_vx, inst_vy # Use instantaneous if no history
            elif global_id in self.global_id_velocity:
                # If it's the same frame or error, return previous velocity
                 vx, vy = self.global_id_velocity[global_id]

        # Store the updated smoothed velocity
        self.global_id_velocity[global_id] = (vx, vy)
        return (vx, vy)

    def _predict_position(self, last_position, velocity, frames_elapsed):
        """Predict new position based on last known position and velocity."""
        if not velocity or frames_elapsed <= 0:
             return last_position # Cannot predict

        vx, vy = velocity
        # Simple linear prediction: pos = last_pos + velocity * time
        predicted_x = last_position[0] + vx * frames_elapsed
        predicted_y = last_position[1] + vy * frames_elapsed
        return (predicted_x, predicted_y)

    def _run_tracker(self, frame):
        """Helper method to run YOLO tracking."""
        return self.yolo.track(
                source=frame,
                conf=self.conf,
                persist=True,
                classes=0, # Only track class 0 (person)
                verbose=False,
                tracker=self.tracker_cfg
            )

    def _extract_tracker_results(self, results):
        """Helper method to extract data from tracker results, filtering invalid IDs."""
        res = results[0] if results else None
        if not res or not hasattr(res, 'boxes') or res.boxes is None or len(res.boxes) == 0:
            return [], [], [], [] # No detections

        try:
            xyxys = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()

            if hasattr(res.boxes, 'id') and res.boxes.id is not None:
                track_ids_raw = res.boxes.id.cpu().numpy()
                # Filter out detections that lost their ID (assign -1 temporarily)
                track_ids_int = [int(tid) if tid is not None else -1 for tid in track_ids_raw]
                valid_indices = [i for i, tid in enumerate(track_ids_int) if tid != -1]

                if len(valid_indices) < len(track_ids_int):
                     logger.warning(f"Frame {self.frame_idx}: {len(track_ids_int) - len(valid_indices)} track(s) missing IDs. Filtering.")

                # Filter all data based on valid indices
                xyxys_f = xyxys[valid_indices]
                confs_f = confs[valid_indices]
                track_ids_f = [track_ids_int[i] for i in valid_indices]
                # original_indices needed to map back to features later
                original_indices_f = valid_indices
            else:
                logger.warning(f"Frame {self.frame_idx}: No track IDs returned by tracker. Cannot associate tracks.")
                return [], [], [], [] # Cannot proceed without track IDs

            return xyxys_f, confs_f, track_ids_f, original_indices_f

        except Exception as e:
             logger.error(f"Error extracting tracker results: {e}\n{traceback.format_exc()}")
             return [], [], [], []


    def _prepare_crops(self, frame, xyxys):
        """Helper method to prepare crops for ReID."""
        crops = []
        valid_crop_indices = [] # Indices relative to the input xyxys
        h_frame, w_frame = frame.shape[:2]

        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            # Clamp coordinates
            x1i, y1i = max(0, x1i), max(0, y1i)
            x2i, y2i = min(w_frame, x2i), min(h_frame, y2i)

            crop_h, crop_w = y2i - y1i, x2i - x1i
            # Increased minimum size slightly
            if crop_w > 10 and crop_h > 10:
                try:
                    crop = frame[y1i:y2i, x1i:x2i]
                    if crop.size > 0:
                        crops.append(crop)
                        valid_crop_indices.append(i) # Store index relative to input xyxys
                    else:
                         logger.warning(f"Frame {self.frame_idx}: Generated empty crop for bbox {i} [{x1i},{y1i},{x2i},{y2i}].")
                except Exception as crop_err:
                    logger.error(f"Frame {self.frame_idx}: Error cropping bbox {i} [{x1i},{y1i},{x2i},{y2i}]: {crop_err}")
            # else: # Debug log removed for brevity
                # logger.debug(f"Frame {self.frame_idx}: Skipping crop for small/invalid bbox {i} [{x1i},{y1i},{x2i},{y2i}]")

        return crops, valid_crop_indices

    def _associate_ocr_iou(self, xyxys, jersey_detections, original_indices):
        """Associate OCR detections with player boxes using IoU and spatial constraints."""
        player_jerseys = {}  # index -> (jersey_num, confidence)
        if not jersey_detections or len(xyxys) == 0:
            return player_jerseys

        assigned_jersey_indices = set()
        
        for i, player_bbox in enumerate(xyxys):
            original_idx = original_indices[i]
            best_iou = self.iou_match_threshold
            best_match_idx = -1
            best_jersey_info = None
            best_spatial_score = 0.0  # IMPROVEMENT: Track spatial score
            
            # IMPROVEMENT: Define the upper body region of the player
            # Jersey numbers are typically on the upper part of the player
            x1, y1, x2, y2 = player_bbox
            upper_y1 = y1
            upper_y2 = y1 + (y2 - y1) * 0.6  # Upper 60% of player box
            upper_body_bbox = [x1, upper_y1, x2, upper_y2]
            
            for j, jersey_det in enumerate(jersey_detections):
                if j in assigned_jersey_indices:
                    continue
                    
                jersey_bbox = jersey_det['bbox']
                
                # IMPROVEMENT: Calculate overlap with upper body region
                # This ensures jersey is detected where it should be
                upper_iou = self.ocr.calculate_iou(upper_body_bbox, jersey_bbox)
                full_iou = self.ocr.calculate_iou(player_bbox, jersey_bbox)
                
                # IMPROVEMENT: Prefer jerseys in upper body region
                spatial_score = upper_iou * 1.5  # Boost upper body matches
                
                # Use regular IoU for threshold check, but spatial score for ranking
                if full_iou > best_iou and spatial_score > best_spatial_score:
                    best_iou = full_iou
                    best_spatial_score = spatial_score
                    best_match_idx = j
                    best_jersey_info = (jersey_det['number'], jersey_det['confidence'])
            
            if best_match_idx != -1:
                assigned_jersey_indices.add(best_match_idx)
                player_jerseys[original_idx] = best_jersey_info
                
        return player_jerseys

    def _add_to_temporal_buffer(self, frame_idx, objects):
        """
        Add unmatched objects to temporal buffer and maintain its size.
        
        Args:
            frame_idx (int): Current frame index
            objects (list): List of unmatched objects with features
        """
        if not objects:
            return
            
        # Add to buffer
        self.unmatched_temporal_buffer.append((frame_idx, objects))
        
        # Keep buffer size limited
        while len(self.unmatched_temporal_buffer) > self.max_buffer_frames:
            # Remove oldest entries
            self.unmatched_temporal_buffer.pop(0)

    def _match_inactive(self, tid, reid_feat, center_pos, jersey_num, forbidden_gids, frame_shape):
        """Helper method to match against inactive tracks with motion prediction."""
        best_inactive_match_gid = -1
        # Slightly lower threshold for inactive matching to be more lenient
        best_inactive_score = 0.75

        h_frame, w_frame = frame_shape[:2]
        max_dist = np.sqrt(w_frame**2 + h_frame**2)

        inactive_candidates = list(self.inactive_tracks.items())

        for inactive_gid, inactive_data in inactive_candidates:
            if inactive_gid in forbidden_gids: continue

            frames_inactive = self.frame_idx - inactive_data['last_frame']
            if frames_inactive <= 0 or frames_inactive > self.temporal_window: continue # Invalid frame diff or too old

            # Motion prediction
            last_pos = inactive_data['last_position']
            velocity = inactive_data.get('velocity', (0, 0))
            predicted_pos = self._predict_position(last_pos, velocity, frames_inactive)

            # Distances
            direct_distance = np.linalg.norm(np.array(center_pos) - np.array(last_pos))
            predicted_distance = np.linalg.norm(np.array(center_pos) - np.array(predicted_pos))
            effective_distance = direct_distance * (1 - self.motion_prediction_weight) + predicted_distance * self.motion_prediction_weight

            # Spatial Proximity Score
            # Increase allowed distance for longer absences, max ~50% diagonal
            distance_threshold = max_dist * min(0.5, 0.2 + 0.01 * frames_inactive)
            spatial_proximity = max(0.0, 1.0 - (effective_distance / distance_threshold))
            if spatial_proximity < 0.1: continue # Quickly prune unlikely spatial matches

            # Temporal Proximity Score
            temporal_proximity = max(0.0, 1.0 - (frames_inactive / self.temporal_window))

            # Appearance Similarity Score - using clustered similarity if available
            appearance_sim = 0.0
            if inactive_data.get('features'):
                try:
                    if self.reid.use_clustering and inactive_gid in self.reid.global_id_clusters:
                        appearance_sim = self.reid.compute_clustered_similarity(reid_feat, inactive_gid)
                    else:
                        valid_inactive_feats = [f for f in inactive_data['features'] if isinstance(f, np.ndarray) and f.size > 0]
                        if valid_inactive_feats:
                            avg_inactive_feat = np.mean(np.stack(valid_inactive_feats), axis=0)
                            appearance_sim = self.reid.compute_similarity(reid_feat, avg_inactive_feat)
                except (ValueError, TypeError): pass

            # Get fused score using jersey information
            inactive_jersey = inactive_data.get('jersey_num')
            jersey_score = self.reid.compute_jersey_score(jersey_num, 
                                                        0.6 if jersey_num is not None else 0.0, 
                                                        inactive_gid)
            
            # Compute fused score with weights
            combined_score = (self.reid.reid_weight * appearance_sim + 
                              self.reid.jersey_weight * jersey_score +
                              0.2 * spatial_proximity +    
                              0.1 * temporal_proximity)

            if combined_score > best_inactive_score:
                best_inactive_score = combined_score
                best_inactive_match_gid = inactive_gid

        # If a good match found
        if best_inactive_match_gid != -1:
            gid = best_inactive_match_gid
            logger.info(f"Frame {self.frame_idx}: Reappeared Track! Matched TID {tid} to inactive GID {gid} (Score: {best_inactive_score:.3f}).")
            self.inactive_tracks.pop(gid, None) # Remove from inactive
            self._reappearances_matched += 1
            return gid
        else:
            return None # No suitable inactive match found

    def _update_inactive_list(self, active_global_ids):
        """Manages the inactive_tracks list and updates player_tracking_data for visibility."""

        # 1. Identify tracks that just became inactive
        newly_inactive_data = [] # Store (gid, jersey_num, last_pos)
        for gid, last_seen_data in list(self.global_id_last_seen.items()):
             if last_seen_data['frame'] == self.frame_idx - 1 and gid not in active_global_ids:
                 if gid not in self.inactive_tracks:
                     recent_features = list(self.reid.global_id_features.get(gid, deque()))[-15:]
                     if not recent_features: continue

                     jersey_num, _ = self.reid.get_best_jersey(gid)
                     velocity = self.global_id_velocity.get(gid, (0.0, 0.0)) # Get last known velocity

                     self.inactive_tracks[gid] = {
                         'last_position': last_seen_data['position'],
                         'last_frame': last_seen_data['frame'],
                         'features': recent_features,
                         'jersey_num': jersey_num,
                         'velocity': velocity
                     }
                     newly_inactive_data.append((gid, jersey_num, last_seen_data['position']))
                     logger.debug(f"Frame {self.frame_idx}: GID {gid} marked as inactive.")

        # 2. Update player_tracking_data for newly inactive players
        for gid, jersey_num, last_pos in newly_inactive_data:
            timestamp = self.frame_idx / self.fps

            # Determine the key for player_tracking_data
            tracking_key = jersey_num if jersey_num is not None else f"gid_{gid}"

            # Add an entry marking the player as invisible
            # Avoid adding if the list is empty or last entry already invisible
            if self.player_tracking_data[tracking_key] and \
               not self.player_tracking_data[tracking_key][-1].get('currently_visible', True):
               continue # Already marked as invisible

            self.player_tracking_data[tracking_key].append({
                'frame': self.frame_idx, # Invisibility starts *now*
                'time': timestamp,
                'global_id': gid,
                'position': last_pos,
                'bbox': None,
                'jersey': jersey_num,
                'jersey_conf': 0.0,
                'currently_visible': False,
                'current_jersey_detection': False
            })
            logger.debug(f"Frame {self.frame_idx}: Marked player key '{tracking_key}' (GID {gid}) as invisible.")


        # 3. Clean up tracks inactive for too long
        inactive_to_remove = [gid for gid, data in self.inactive_tracks.items()
                              if self.frame_idx - data['last_frame'] > self.temporal_window]

        if inactive_to_remove:
             logger.debug(f"Frame {self.frame_idx}: Removing {len(inactive_to_remove)} GIDs from inactive list (timeout: {self.temporal_window} frames). GIDs: {inactive_to_remove}")
             for gid in inactive_to_remove:
                 self.inactive_tracks.pop(gid, None)
                 # Also remove velocity for timed-out tracks
                 self.global_id_velocity.pop(gid, None)

    def _final_jersey_conflict_resolution(self, annotated_objs, player_jerseys, original_indices, track_ids_map):
        """Helper method for jersey conflict resolution after all assignments."""
        jersey_conflict_map = defaultdict(list)
        high_conf = self.ocr.high_conf_threshold # Use threshold from OCR instance

        for idx, obj in enumerate(annotated_objs):
             current_jersey_num = obj.get("current_jersey_detection")
             if current_jersey_num is None: continue

             # Find original confidence for this detection
             current_jersey_conf = 0.0
             tid = obj["track_id"]
             if tid in track_ids_map:
                 original_idx = track_ids_map[tid] # Get original index before filtering
                 if original_idx in player_jerseys:
                      num, conf = player_jerseys[original_idx]
                      if num == current_jersey_num:
                          current_jersey_conf = conf

             # Check if confidence meets the high threshold
             if current_jersey_conf >= high_conf:
                  jersey_conflict_map[current_jersey_num].append(
                      (obj["global_id"], current_jersey_conf, idx) # Store GID, conf, index in annotated_objs
                  )

        # Resolve conflicts
        ids_merged_in_final_pass = set()
        for jersey_num, assignments in jersey_conflict_map.items():
            if len(assignments) > 1: # Conflict: Same high-conf jersey on multiple players
                assignments.sort(key=lambda x: x[1], reverse=True) # Highest confidence first
                correct_gid, correct_conf, correct_idx = assignments[0]

                logger.warning(f"Frame {self.frame_idx}: Final Jersey Conflict! Jersey #{jersey_num} detected on GIDs: {[(g,f'{c:.2f}') for g,c,_ in assignments]}. Merging into GID {correct_gid} (Conf: {correct_conf:.2f}).")

                for other_gid, other_conf, other_idx in assignments[1:]:
                    if other_gid != correct_gid and other_gid not in ids_merged_in_final_pass:
                        # --- Merge Velocities (Before ReID Merge affects velocity dict) ---
                        merged_vx, merged_vy = 0.0, 0.0
                        if other_gid in self.global_id_velocity and correct_gid in self.global_id_velocity:
                            v_other = self.global_id_velocity[other_gid]
                            v_correct = self.global_id_velocity[correct_gid]
                            # Weighted average (e.g., 80% correct, 20% other)
                            merged_vx = 0.8 * v_correct[0] + 0.2 * v_other[0]
                            merged_vy = 0.8 * v_correct[1] + 0.2 * v_other[1]
                            logger.debug(f"Merging velocity for {other_gid} -> {correct_gid}. New vel: ({merged_vx:.2f}, {merged_vy:.2f})")
                        elif correct_gid in self.global_id_velocity:
                            merged_vx, merged_vy = self.global_id_velocity[correct_gid] # Keep correct GID's velocity
                        elif other_gid in self.global_id_velocity:
                             merged_vx, merged_vy = self.global_id_velocity[other_gid] # Use other GID's velocity

                        # Update velocity of target GID
                        if merged_vx != 0.0 or merged_vy != 0.0:
                             self.global_id_velocity[correct_gid] = (merged_vx, merged_vy)
                        # Remove velocity of source GID
                        self.global_id_velocity.pop(other_gid, None)
                        # --- End Velocity Merge ---

                        # Perform the merge in the ReID tracker
                        self.reid._merge_global_ids(source_gid=other_gid, target_gid=correct_gid)
                        ids_merged_in_final_pass.add(other_gid)

                        # Update trackid_to_globalid map for all TIDs affected
                        tids_to_update = [tid for tid, mapped_gid in self.trackid_to_globalid.items() if mapped_gid == other_gid]
                        for tid_to_update in tids_to_update:
                            self.trackid_to_globalid[tid_to_update] = correct_gid
                            logger.info(f"Updated TID {tid_to_update} mapping post-merge: {other_gid} -> {correct_gid}")

                        # Update annotated_objs for the current frame
                        if other_idx < len(annotated_objs) and annotated_objs[other_idx]["global_id"] == other_gid:
                             annotated_objs[other_idx]["global_id"] = correct_gid
                             # Re-fetch voted jersey info based on merged GID
                             new_voted_jersey, new_voted_conf = self.reid.get_best_jersey(correct_gid)
                             annotated_objs[other_idx]["voted_jersey_num"] = new_voted_jersey
                             annotated_objs[other_idx]["voted_jersey_conf"] = new_voted_conf
                             # Update velocity info in annotated object as well
                             annotated_objs[other_idx]["velocity"] = (merged_vx, merged_vy)

                        # Update the 'correct' GID's entry in case its voted jersey changed
                        if correct_idx < len(annotated_objs) and annotated_objs[correct_idx]["global_id"] == correct_gid:
                             new_voted_jersey_corr, new_voted_conf_corr = self.reid.get_best_jersey(correct_gid)
                             annotated_objs[correct_idx]["voted_jersey_num"] = new_voted_jersey_corr
                             annotated_objs[correct_idx]["voted_jersey_conf"] = new_voted_conf_corr
                             annotated_objs[correct_idx]["velocity"] = (merged_vx, merged_vy) # Update correct GID velocity too

                        # Clean up auxiliary tracking structures
                        self.global_id_last_seen.pop(other_gid, None)
                        self.inactive_tracks.pop(other_gid, None)
                        self.track_history_viz.pop(other_gid, None)

                        # Merge player_tracking_data entries
                        other_key = f"gid_{other_gid}"
                        source_data = self.player_tracking_data.pop(other_key, [])
                        if source_data: # If there was data for the non-jersey ID
                             new_voted_jersey_corr, _ = self.reid.get_best_jersey(correct_gid)
                             # Determine target key based on potentially new jersey
                             target_key = new_voted_jersey_corr if new_voted_jersey_corr is not None else f"gid_{correct_gid}"
                             for entry in source_data:
                                 entry['global_id'] = correct_gid # Update GID in entry
                                 if new_voted_jersey_corr is not None: # Update jersey if available now
                                     entry['jersey'] = new_voted_jersey_corr
                                 self.player_tracking_data[target_key].append(entry)
                             # Re-sort the target list after appending (optional but good practice)
                             self.player_tracking_data[target_key].sort(key=lambda x: x['frame'])
                             logger.info(f"Merged player_tracking_data from '{other_key}' into '{target_key}'")


        return annotated_objs

    def process_frame(self, frame):
        """
        Process a single video frame: Track -> OCR -> Associate -> Annotate.
        Handles temporal consistency and collects highlight data.

        Args:
            frame (numpy.ndarray): Input video frame (BGR format)

        Returns:
            tuple: (annotated_frame, track_data_list)
                   annotated_frame (numpy.ndarray): Frame with visualizations.
                   track_data_list (list): List of dictionaries, one per tracked object.
        """
        process_start = time.time()
        self.frame_idx += 1
        h_frame, w_frame = frame.shape[:2]

        # --- Run YOLO + Tracker ---
        try:
            results = self._run_tracker(frame)
            # xyxys_filt, confs_filt, track_ids_filt, original_indices
            xyxys, confs, track_ids, original_indices = self._extract_tracker_results(results)

            if not track_ids:
                logger.debug(f"Frame {self.frame_idx}: No valid tracks found.")
                self._update_inactive_list(set())
                process_time = time.time() - process_start
                self.processing_times.append(process_time)
                self.processed_frames += 1
                return frame, []

            # Create a map from TID to original index for easy lookup later
            track_ids_map = {tid: original_idx for tid, original_idx in zip(track_ids, original_indices)}

        except Exception as e:
            logger.error(f"Error during tracking/extraction (frame {self.frame_idx}): {str(e)}\n{traceback.format_exc()}")
            self._update_inactive_list(set())
            process_time = time.time() - process_start
            self.processing_times.append(process_time)
            self.processed_frames += 1
            return frame, []

        # --- STEP 1: Whole Frame OCR (at interval) ---
        jersey_detections = []
        if self.frame_idx % self.ocr_interval == 0:
            try:
                ocr_start = time.time()
                jersey_detections = self.ocr.detect_whole_frame(frame)
                logger.debug(f"Frame {self.frame_idx}: OCR found {len(jersey_detections)} potential jerseys (took {time.time()-ocr_start:.3f}s).")
            except Exception as e:
                logger.error(f"Error during whole-frame OCR (frame {self.frame_idx}): {str(e)}\n{traceback.format_exc()}")

        # --- STEP 2: Associate OCR Detections with Player Boxes via IoU ---
        # player_jerseys maps *original_index* -> (jersey_num, confidence)
        player_jerseys = self._associate_ocr_iou(xyxys, jersey_detections, original_indices)
        if player_jerseys:
             logger.debug(f"Frame {self.frame_idx}: Associated {len(player_jerseys)} jerseys via IoU.")

        # --- STEP 3: Prepare Crops and Extract ReID Features ---
        # Prepare crops ONLY for the filtered detections (xyxys)
        try:
            reid_start = time.time()
            # Pass the filtered xyxys to prepare_crops
            crops, valid_crop_indices_rel_filt = self._prepare_crops(frame, xyxys)

            if not crops:
                logger.warning(f"Frame {self.frame_idx}: No valid crops generated from filtered detections. Skipping ReID/Association.")
                self._update_inactive_list(set())
                process_time = time.time() - process_start
                self.processing_times.append(process_time)
                self.processed_frames += 1
                annotated_frame = self.annotate(frame.copy(), [])
                return annotated_frame, []

            features_list = self.reid.extract_features(crops)
            # Map features back to the *filtered* track_ids indices using valid_crop_indices_rel_filt
            features = {} # Map: index_in_filtered_lists -> feature
            for k, feat in enumerate(features_list):
                filt_idx = valid_crop_indices_rel_filt[k] # Index relative to filtered lists (xyxys, confs, track_ids)
                features[filt_idx] = feat

            logger.debug(f"Frame {self.frame_idx}: Extracted {len(features)} ReID features (took {time.time()-reid_start:.3f}s).")

        except Exception as e:
            logger.error(f"Error preparing crops or extracting ReID (frame {self.frame_idx}): {str(e)}\n{traceback.format_exc()}")
            self._update_inactive_list(set())
            process_time = time.time() - process_start
            self.processing_times.append(process_time)
            self.processed_frames += 1
            return frame, []

        # --- STEP 4: Match Detections to Global IDs (Main Logic) ---
        association_start = time.time()
        current_frame_tids_set = set(track_ids)
        active_global_ids = set()
        annotated_objs = []
        assigned_gids_in_frame = set()
        
        # Prepare buffer for unmatched detections this frame
        unmatched_buffer_this_frame = []

        # Iterate through the *filtered* detections (indices 0 to len(track_ids)-1)
        for i, tid in enumerate(track_ids):
            if i not in features: # Check if feature exists for this filtered index
                logger.warning(f"Frame {self.frame_idx}: No feature for filtered index {i} (TID {tid}), skipping association.")
                continue

            try:
                x1, y1, x2, y2 = xyxys[i]
                reid_feat = features[i]
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                current_pos = (center_x, center_y)
                current_bbox = [int(x1), int(y1), int(x2), int(y2)]

                # Get jersey info using the original index
                original_idx = original_indices[i]
                jersey_num, jersey_conf = player_jerseys.get(original_idx, (None, 0.0))

                gid = -1

                # --- Temporal/Spatial Matching for Reappearance ---
                matched_inactive = False
                if tid not in self.trackid_to_globalid:
                    inactive_match_gid = self._match_inactive(tid, reid_feat, current_pos, jersey_num, assigned_gids_in_frame, frame.shape)
                    if inactive_match_gid is not None:
                        gid = inactive_match_gid
                        self.trackid_to_globalid[tid] = gid
                        matched_inactive = True
                        assigned_gids_in_frame.add(gid) # Mark as assigned in this frame
                        # Add feature to ReID tracker
                        self.reid.global_id_features[gid].append(reid_feat)
                        if jersey_num is not None and jersey_conf > 0.5:
                            self.reid.global_id_jersey_votes[gid][jersey_num] += 1
                            self.reid._update_jersey_mapping(gid)
                            if (gid, jersey_num) not in self.reid.jersey_first_seen:
                                self.reid.jersey_first_seen[(gid, jersey_num)] = self.frame_idx

                # --- Standard ReID Matching ---
                if not matched_inactive:
                    forbidden_gids = set()
                    for other_tid, assigned_gid in self.trackid_to_globalid.items():
                        if other_tid != tid and other_tid in current_frame_tids_set:
                            forbidden_gids.add(assigned_gid)
                    forbidden_gids.update(assigned_gids_in_frame)

                    if tid not in self.trackid_to_globalid:
                        gid = self.reid.match_to_global_id(
                            reid_feat, forbidden_gids, jersey_num, jersey_conf, 
                            self.frame_idx, current_bbox
                        )
                        self.trackid_to_globalid[tid] = gid
                        assigned_gids_in_frame.add(gid) # Mark as assigned
                    else:
                        gid = self.trackid_to_globalid[tid]
                        if gid in assigned_gids_in_frame and gid != self.trackid_to_globalid.get(tid):
                             logger.warning(f"Frame {self.frame_idx}: GID {gid} conflict for TID {tid}. Already assigned. Investigate.")
                             # Decide how to handle - skip this track? Force new ID? For now, skip maybe.
                             continue

                        assigned_gids_in_frame.add(gid) # Ensure it's marked

                        # Add to ReID feature history
                        self.reid.global_id_features[gid].append(reid_feat)
                        if jersey_num is not None and jersey_conf > 0.5:
                            self.reid.global_id_jersey_votes[gid][jersey_num] += 1
                            self.reid._update_jersey_mapping(gid)
                            if (gid, jersey_num) not in self.reid.jersey_first_seen:
                                self.reid.jersey_first_seen[(gid, jersey_num)] = self.frame_idx

                        # Check for high-confidence jersey override/merge
                        if jersey_num is not None and jersey_conf >= self.ocr.high_conf_threshold:
                             merged = self.reid.override_with_jersey(gid, jersey_num, jersey_conf, self.frame_idx)
                             if merged:
                                 # Need to update mapping for ALL TIDs that were pointing to the old gid
                                 # Since jersey merges can change the GID mapping
                                 best_jersey, _ = self.reid.get_best_jersey(gid)
                                 if best_jersey == jersey_num:
                                     # The jersey was accepted, make sure mapping is correct
                                     for t, g in list(self.trackid_to_globalid.items()):
                                        if g == gid:
                                            # GID is updated (should be good)
                                            pass

                # --- Final Processing ---
                if gid != -1:
                    active_global_ids.add(gid)
                    
                    # Calculate velocity
                    velocity = self._calculate_velocity(gid, current_pos)
                    # Store velocity (will be used in next frame's prediction)
                    # self.global_id_velocity[gid] = velocity # Already done in _calculate_velocity

                    # Update last seen info
                    self.global_id_last_seen[gid] = {'frame': self.frame_idx, 'position': current_pos}

                    # Get voted jersey
                    best_voted_jersey, best_voted_conf = self.reid.get_best_jersey(gid)

                    # Prepare data for annotation / export
                    track_data = {
                        "frame_idx": self.frame_idx, "track_id": tid, "global_id": gid,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)], "score": float(confs[i]),
                        "center": current_pos, "current_jersey_detection": jersey_num,
                        "current_jersey_confidence": float(jersey_conf),
                        "voted_jersey_num": best_voted_jersey, "voted_jersey_conf": float(best_voted_conf),
                        "velocity": velocity # Include velocity
                    }

                    annotated_objs.append(track_data)
                    
                    # Update viz history
                    self.track_history_viz[gid].append({
                         "frame": self.frame_idx, "x": center_x, "y": center_y,
                         "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })

                    # --- Update player_tracking_data (for ALL players) ---
                    tracking_key = best_voted_jersey if best_voted_jersey is not None else f"gid_{gid}"
                    timestamp = self.frame_idx / self.fps

                    track_entry = {
                        'frame': self.frame_idx, 'time': timestamp, 'global_id': gid, 'track_id': tid,
                        'position': current_pos, 'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'jersey': best_voted_jersey, 'jersey_conf': float(best_voted_conf),
                        'currently_visible': True, 'current_jersey_detection': jersey_num is not None,
                        'current_jersey_conf': float(jersey_conf)
                    }
                    self.player_tracking_data[tracking_key].append(track_entry)
                    
                else:
                    # Track was not matched - add to buffer for temporal matching
                    unmatched_buffer_this_frame.append({
                        'feature': reid_feat,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'jersey_num': jersey_num,
                        'jersey_conf': float(jersey_conf),
                        'center': current_pos
                    })

            except Exception as e:
                 logger.error(f"Error processing track TID {tid} at filtered index {i} (Frame {self.frame_idx}): {str(e)}\n{traceback.format_exc()}")
                 continue # Skip to next detection

        # --- Add unmatched detections to ReID buffer for future matching ---
        for obj in unmatched_buffer_this_frame:
            self.reid.add_to_temporal_buffer(obj['feature'], self.frame_idx, obj['bbox'])

        logger.debug(f"Frame {self.frame_idx}: Association done (took {time.time()-association_start:.3f}s). Active GIDs: {len(active_global_ids)}")
        
        # --- STEP 5: Final Jersey Conflict Resolution ---
        annotated_objs = self._final_jersey_conflict_resolution(annotated_objs, player_jerseys, original_indices, track_ids_map)

        # --- STEP 6: Update Inactive Tracks List ---
        self._update_inactive_list(active_global_ids)

        # --- STEP 7: Annotate Frame ---
        try:
            annotated_frame = self.annotate(frame.copy(), annotated_objs, show_trails=False)
        except Exception as e:
            logger.error(f"Error during frame annotation (frame {self.frame_idx}): {str(e)}\n{traceback.format_exc()}")
            annotated_frame = frame.copy() # Use original frame on error

        # --- Finalize Frame Processing ---
        process_time = time.time() - process_start
        self.processing_times.append(process_time)
        self.processed_frames += 1
        
        if self.frame_idx % 50 == 0:
             logger.info(f"Frame {self.frame_idx}: Process Time: {process_time:.4f}s, Active GIDs: {len(active_global_ids)}, Inactive: {len(self.inactive_tracks)}")

        return annotated_frame, annotated_objs

    def annotate(self, frame, tracks, show_trails=False, trail_length=30):
        """Draw bounding boxes, IDs, jersey numbers, and optionally trails."""
        # --- Trail Drawing ---
        if show_trails:
            for t in tracks:
                gid = t['global_id']
                history = list(self.track_history_viz[gid])
                if len(history) < 2: continue
                history = history[-trail_length:]
                points = [(int(h['x']), int(h['y'])) for h in history if h.get('x') is not None and h.get('y') is not None]
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    try:
                       cv2.line(frame, points[i-1], points[i], (0, int(165*alpha), int(255*alpha)), 2)
                    except (OverflowError, TypeError):
                       cv2.line(frame, points[i-1], points[i], (0, 165, 255), 2)

        # --- Color Scheme & Font ---
        def get_player_color(gid, voted_jersey_num=None):
            """Get a consistent color for a player based on jersey number or GID."""
            if voted_jersey_num is not None:
                h = (voted_jersey_num * 137) % 180
                s = 200; v = 255
                try:
                    bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
                    return tuple(bgr.tolist())
                except Exception: pass # Fallback on error
            # Fallback: Consistent color based on global ID
            np.random.seed(gid)
            return tuple(np.random.randint(0, 256, 3).tolist())
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_color = (255, 255, 255) # White text

        # --- Frame Counter ---
        cv2.putText(frame, f"Frame: {self.frame_idx}", (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # --- Draw Players ---
        for t in tracks:
            try:
                x1, y1, x2, y2 = t["bbox"]
                gid = t["global_id"]
                voted_jersey = t.get("voted_jersey_num")
                current_jersey = t.get("current_jersey_detection")
                current_jersey_conf = t.get("current_jersey_confidence", 0.0)

                color = get_player_color(gid, voted_jersey)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Label preparation
                label = ""
                if voted_jersey is not None:
                    label = f"#{voted_jersey}"
                    if current_jersey == voted_jersey and current_jersey_conf >= self.ocr.high_conf_threshold:
                         label += "*"
                else:
                    label = f"ID:{gid}" # Show GID if no jersey

                # Draw label background and text
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                label_y = max(y1, text_h + 10)
                bg_x1, bg_y1 = x1, label_y - text_h - baseline - 2
                bg_x2, bg_y2 = x1 + text_w + 4, label_y + baseline - 2
                bg_y1 = max(0, bg_y1)

                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                cv2.putText(frame, label, (x1 + 2, label_y - baseline // 2 - 2 ), font, font_scale, text_color, thickness, cv2.LINE_AA)

            except Exception as annotate_err:
                 logger.error(f"Error annotating track {t.get('global_id', 'N/A')}: {annotate_err}")
                 continue

        return frame

    def generate_player_highlight_data(self, output_path=None, min_segment_duration=1.0, min_total_duration=3.0):
        """
        Generates structured highlight information for ALL tracked players (jersey or GID based).

        Args:
            output_path (str, optional): Path to save the highlight data as JSON.
            min_segment_duration (float): Min duration (seconds) for a segment.
            min_total_duration (float): Min total duration (seconds) for a player.

        Returns:
            dict: Highlight data {tracking_key: player_info_dict}.
                  tracking_key is jersey_num (int) or "gid_{gid}" (str).
        """
        logger.info("Generating highlight data for all tracked players...")
        highlight_data = {}

        # Iterate through all keys (jersey numbers and "gid_X")
        for tracking_key, track_entries in self.player_tracking_data.items():
            if not track_entries: continue

            # Determine player identifier (jersey or GID) for logging/output
            player_id_str = f"Player #{tracking_key}" if isinstance(tracking_key, int) else f"Player {tracking_key}"
            logger.debug(f"Processing {player_id_str}, found {len(track_entries)} entries.")

            sorted_entries = sorted(track_entries, key=lambda x: x['frame'])
            segments = []
            current_segment = None

            # Group into segments (same logic as before)
            for entry in sorted_entries:
                is_visible = entry.get('currently_visible', False)
                if is_visible:
                    if current_segment is None:
                        current_segment = {'start_frame': entry['frame'], 'start_time': entry['time'], 'entries': [entry]}
                    else:
                        last_entry = current_segment['entries'][-1]
                        if entry['frame'] - last_entry['frame'] <= self.fps * 0.5: # Allow ~0.5s gap
                            current_segment['entries'].append(entry)
                        else:
                            current_segment['end_frame'] = last_entry['frame']
                            current_segment['end_time'] = last_entry['time']
                            segments.append(current_segment)
                            current_segment = {'start_frame': entry['frame'], 'start_time': entry['time'], 'entries': [entry]}
                else: # Invisibility marker
                    if current_segment is not None:
                        last_entry = current_segment['entries'][-1]
                        current_segment['end_frame'] = last_entry['frame']
                        current_segment['end_time'] = last_entry['time']
                        segments.append(current_segment)
                        current_segment = None

            if current_segment is not None and current_segment['entries']:
                 last_entry = current_segment['entries'][-1]
                 current_segment['end_frame'] = last_entry['frame']
                 current_segment['end_time'] = last_entry['time']
                 segments.append(current_segment)

            # Process and filter segments (same logic as before)
            processed_segments = []
            total_duration_player = 0.0
            for segment in segments:
                 # Ensure start/end times are valid before calculating duration
                 start_t = segment.get('start_time', 0.0)
                 end_t = segment.get('end_time', start_t) # Default end to start if missing
                 duration_sec = max(0.0, end_t - start_t) # Ensure non-negative

                 if duration_sec < min_segment_duration: continue

                 visible_entries = segment.get('entries', [])
                 if not visible_entries: continue

                 jersey_detected_count = sum(1 for e in visible_entries if e.get('current_jersey_detection', False))
                 jersey_visibility_perc = (jersey_detected_count / len(visible_entries)) * 100 if visible_entries else 0

                 track_coords = [{'frame': e['frame'], 'time': e['time'], 'position': e.get('position'), 'bbox': e.get('bbox')}
                                 for e in visible_entries if e.get('bbox') is not None]

                 processed_segment = {
                     'start_frame': segment['start_frame'], 'end_frame': segment['end_frame'],
                     'start_time': start_t, 'end_time': end_t, 'duration_sec': duration_sec,
                     'visible_frames': len(visible_entries), 'jersey_detected_frames': jersey_detected_count,
                     'jersey_visibility_perc': jersey_visibility_perc, 'track_coords': track_coords
                 }
                 processed_segments.append(processed_segment)
                 total_duration_player += duration_sec

            # Add player data if valid segments exist and meet total duration
            if processed_segments and total_duration_player >= min_total_duration:
                # Use the original tracking_key (int or str) for the output dictionary
                output_key = tracking_key # Could be int (jersey) or str ("gid_X")
                highlight_data[output_key] = {
                    'identifier': output_key, # Store the key itself for clarity
                    'segments': processed_segments,
                    'total_segments': len(processed_segments),
                    'total_duration_sec': total_duration_player,
                    'average_segment_duration': total_duration_player / len(processed_segments)
                }
                logger.info(f"Generated highlight data for {player_id_str}: {len(processed_segments)} segments, {total_duration_player:.1f}s total.")
            else:
                 logger.info(f"Skipping {player_id_str}: Insufficient valid segment duration ({total_duration_player:.1f}s < {min_total_duration:.1f}s).")

        # Save to file
        if output_path and highlight_data:
            try:
                class NpEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer): return int(obj)
                        if isinstance(obj, np.floating): return float(obj)
                        if isinstance(obj, np.ndarray): return obj.tolist()
                        return super(NpEncoder, self).default(obj)
                output_dir_path = os.path.dirname(output_path)
                if output_dir_path and not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(highlight_data, f, indent=2, cls=NpEncoder)
                logger.info(f"Player highlight data saved successfully to {output_path}")
            except Exception as e:
                logger.error(f"Error saving player highlight data to {output_path}: {str(e)}\n{traceback.format_exc()}")

        return highlight_data
        
    def generate_trajectory(self, output_path, frame_size=None, selected_gids=None):
        """
        Generates trajectory visualization using viz history.

        Args:
            output_path (str): Path to save the trajectory image.
            frame_size (tuple, optional): (width, height) of the original video frame for scaling. Defaults to (1280, 720).
            selected_gids (list, optional):
                - If None: Generates trajectories for ALL tracked players.
                - If list contains multiple GIDs: Generates trajectories for ONLY those players.
                - If list contains a single GID: Generates trajectory ONLY for that specific player
                (intended for individual player visualization).

        Returns:
            numpy.ndarray or None: The generated trajectory image as a NumPy array, or None on failure.
        """
        # Import plt locally to handle potential ImportError gracefully
        try:
            import matplotlib.pyplot as plt
            has_plt = True
        except ImportError:
            has_plt = False
            logger.warning("Matplotlib not found. Trajectory visualization will use fallback colors.")

        if frame_size is None:
            frame_size = (1280, 720)
            logger.warning(f"Frame size not provided, defaulting to {frame_size}.")

        try:
            w_orig, h_orig = frame_size
            if w_orig <= 0 or h_orig <= 0:
                logger.error(f"Invalid frame_size received: {frame_size}")
                return None

            # --- Canvas Setup ---
            w_viz, h_viz = 1280, 720 # Fixed visualization canvas size
            trajectory_img = np.ones((h_viz, w_viz, 3), dtype=np.uint8) * 255 # White background
            scale_x = w_viz / w_orig
            scale_y = h_viz / h_orig

            # --- Draw Field Background ---
            field_color = (50, 150, 50) # Darker green
            border_color = (200, 200, 200) # Light grey border
            line_color = (230, 230, 230) # Lighter grey lines
            line_thickness = 1
            border_margin = 30

            cv2.rectangle(trajectory_img, (border_margin, border_margin), (w_viz - border_margin, h_viz - border_margin), field_color, -1) # Fill field
            cv2.rectangle(trajectory_img, (border_margin, border_margin), (w_viz - border_margin, h_viz - border_margin), border_color, line_thickness + 1) # Border
            # Center line
            cv2.line(trajectory_img, (w_viz // 2, border_margin), (w_viz // 2, h_viz - border_margin), line_color, line_thickness)
            # Center circle (approximate scaling)
            center_circle_radius_pixels = int(9.15 * scale_y * (w_viz / 105.0)) # Assuming approx 105m field width for scaling radius
            cv2.circle(trajectory_img, (w_viz // 2, h_viz // 2), max(10, center_circle_radius_pixels), line_color, line_thickness)
            # Potentially add goal boxes, penalty areas if needed for better context

            # --- Determine Players to Draw ---
            players_to_show = []
            title = "Player Trajectories"
            is_individual_mode = False

            if selected_gids and len(selected_gids) == 1:
                # *** Individual Player Mode ***
                is_individual_mode = True
                gid_single = selected_gids[0]
                if gid_single in self.track_history_viz:
                    players_to_show = [gid_single]
                    jersey_num, _ = self.reid.get_best_jersey(gid_single)
                    title = f"Trajectory for Player GID: {gid_single}"
                    if jersey_num:
                        title += f" (# {jersey_num})"
                    logger.info(f"Generating individual trajectory for GID {gid_single}")
                else:
                    logger.warning(f"Requested individual GID {gid_single} not found in trajectory history.")
                    # Draw text indicating player not found and return
                    cv2.putText(trajectory_img, f"Player GID {gid_single} Not Found", (50, h_viz // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.imwrite(output_path, trajectory_img) # Save the 'not found' image
                    return trajectory_img # Return the image array

            elif selected_gids:
                # *** Multiple Selected Players Mode ***
                players_to_show = [gid for gid in selected_gids if gid in self.track_history_viz]
                title = f"Trajectories for Selected Players ({len(players_to_show)})"
                if not players_to_show:
                    logger.warning("None of the selected GIDs were found in trajectory history.")
                else:
                    logger.info(f"Generating trajectories for {len(players_to_show)} selected players.")

            else:
                # *** All Players Mode ***
                players_to_show = list(self.track_history_viz.keys())
                title = "All Player Trajectories"
                logger.info(f"Generating trajectories for all {len(players_to_show)} tracked players.")

            # Handle case where no players are ultimately selected/found
            if not players_to_show:
                logger.info("No players to draw trajectories for.")
                cv2.putText(trajectory_img, "No Trajectories to Display", (50, h_viz // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.imwrite(output_path, trajectory_img)
                return trajectory_img

            # --- Color Palette ---
            if has_plt:
                try:
                    # Use a perceptually uniform colormap if available
                    num_colors = max(10, len(players_to_show)) # Ensure enough colors
                    try:
                        color_palette = plt.cm.get_cmap('viridis', num_colors) # Example: viridis
                    except ValueError: # Fallback if cmap name is wrong
                        color_palette = plt.cm.get_cmap('tab10', num_colors)

                    # Function to get BGR color tuple from index
                    get_color = lambda idx: [int(c * 255) for c in color_palette(idx % num_colors)[:3]][::-1] # BGR
                except Exception as plt_err:
                    logger.warning(f"Matplotlib colormap error: {plt_err}. Using fallback colors.")
                    has_plt = False # Revert to fallback

            if not has_plt: # Fallback if matplotlib fails or is not installed
                fallback_colors = [
                    (0, 165, 255), (255, 0, 0), (0, 0, 255), (255, 0, 255),
                    (0, 255, 0), (128, 128, 0), (0, 128, 128), (128, 0, 128),
                    (255, 255, 0), (0, 255, 255), (255, 0, 255), (192, 192, 192)
                ]
                get_color = lambda idx: fallback_colors[idx % len(fallback_colors)]

            # --- Draw Trajectories Loop ---
            for player_idx, gid in enumerate(players_to_show):
                history = list(self.track_history_viz.get(gid, [])) # Use .get for safety
                if len(history) < 2:
                    continue # Need at least two points for a line

                jersey_num, _ = self.reid.get_best_jersey(gid)
                color = get_color(player_idx)

                # Convert history to scaled points, handling potential None or NaN values
                points = []
                for entry in history:
                    x = entry.get('x')
                    y = entry.get('y')
                    if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                        px, py = int(x * scale_x), int(y * scale_y)
                        # Clamp points to be within visualization bounds (plus a small margin)
                        px = np.clip(px, -10, w_viz + 10)
                        py = np.clip(py, -10, h_viz + 10)
                        points.append((px, py))
                    else:
                        points.append(None) # Maintain sequence for line breaks

                if sum(p is not None for p in points) < 2: continue # Skip if fewer than 2 valid points

                # Draw lines connecting consecutive valid points
                for i in range(1, len(points)):
                    if points[i - 1] is not None and points[i] is not None:
                        # Optional: Fade trail color/thickness
                        alpha_factor = (i / len(points)) * 0.8 + 0.2 # Fade from 0.2 to 1.0
                        line_color_fade = tuple(int(c * alpha_factor) for c in color)
                        thickness = max(1, int(2 * alpha_factor)) # Fade thickness slightly
                        cv2.line(trajectory_img, points[i - 1], points[i], line_color_fade, thickness, cv2.LINE_AA)

                # Mark start and end points
                first_valid = next((p for p in points if p is not None), None)
                last_valid = next((p for p in reversed(points) if p is not None), None)

                if first_valid:
                    cv2.circle(trajectory_img, first_valid, 4, color, -1) # Small start circle

                if last_valid:
                    cv2.circle(trajectory_img, last_valid, 6, color, -1) # Slightly larger end circle
                    # Add label near the end point
                    label = f"P:{gid}" + (f" (#{jersey_num})" if jersey_num is not None else "")
                    label_pos_x = last_valid[0] + 8
                    label_pos_y = last_valid[1] + 4
                    # Adjust position if label goes off-screen
                    label_pos_x = min(label_pos_x, w_viz - 60) # Keep space for text
                    label_pos_y = min(max(label_pos_y, 15), h_viz - 5) # Keep within bounds

                    # Add background rectangle for better readability
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    bg_tl = (label_pos_x - 2, label_pos_y - text_h - 2)
                    bg_br = (label_pos_x + text_w + 2, label_pos_y + 2)
                    # Semi-transparent background
                    sub_img = trajectory_img[bg_tl[1]:bg_br[1], bg_tl[0]:bg_br[0]]
                    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
                    trajectory_img[bg_tl[1]:bg_br[1], bg_tl[0]:bg_br[0]] = res

                    cv2.putText(trajectory_img, label, (label_pos_x, label_pos_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA) # Black text

            # --- Add Title and Timestamp ---
            cv2.putText(trajectory_img, title, (border_margin, border_margin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(trajectory_img, f"Generated: {timestamp}", (border_margin, h_viz - border_margin + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA) # Grey timestamp

            # --- Save Image ---
            try:
                # Ensure output directory exists
                output_dir_path = os.path.dirname(output_path)
                if output_dir_path and not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path, exist_ok=True)

                cv2.imwrite(output_path, trajectory_img)
                logger.info(f"Trajectory visualization saved successfully to {output_path}")
                return trajectory_img # Return the image array
            except Exception as e:
                logger.error(f"Error saving trajectory image to {output_path}: {str(e)}\n{traceback.format_exc()}")
                return None

        except Exception as e:
            logger.error(f"Critical error during trajectory generation: {str(e)}\n{traceback.format_exc()}")
            return None

    def export_tracking_data(self, output_path, selected_identifiers=None):
        """
        Exports detailed tracking data for selected players (jersey or GID) to CSV.

        Args:
            output_path (str): Path to save the CSV file.
            selected_identifiers (list, optional): List of jersey numbers (int) or
                                                  GID strings ("gid_X") to export. None for all.
        """
        logger.info(f"Exporting detailed tracking data to {output_path}...")
        if not self.player_tracking_data:
            logger.warning("No player tracking data available to export.")
            return False

        try:
            data_rows = []
            identifiers_to_export = selected_identifiers if selected_identifiers else self.player_tracking_data.keys()

            for key in identifiers_to_export:
                # Ensure key exists before accessing
                if key not in self.player_tracking_data:
                     logger.warning(f"Identifier '{key}' not found in tracking data, skipping.")
                     continue

                track_entries = self.player_tracking_data[key]
                is_gid_key = isinstance(key, str) and key.startswith("gid_")

                for entry in track_entries:
                    pos = entry.get('position', (None, None))
                    bbox = entry.get('bbox') # Can be None

                    row = {
                        'frame': entry.get('frame'), 'time': entry.get('time'),
                        'identifier': key, # Jersey number or "gid_X"
                        'global_id': entry.get('global_id'), 'track_id': entry.get('track_id'),
                        'x': pos[0] if pos else None, 'y': pos[1] if pos else None,
                        'bbox_x1': bbox[0] if bbox else None, 'bbox_y1': bbox[1] if bbox else None,
                        'bbox_x2': bbox[2] if bbox else None, 'bbox_y2': bbox[3] if bbox else None,
                        'visible': entry.get('currently_visible', False),
                        'jersey': entry.get('jersey'), # Can be None
                        'jersey_conf_vote': entry.get('jersey_conf', 0.0),
                        'jersey_detected_now': entry.get('current_jersey_detection', False),
                        'jersey_conf_now': entry.get('current_jersey_conf', 0.0)
                    }
                    data_rows.append(row)

            if data_rows:
                df = pd.DataFrame(data_rows)
                df.sort_values(by=['frame', 'identifier'], inplace=True)
                # Ensure output directory exists
                output_dir_path = os.path.dirname(output_path)
                if output_dir_path and not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path, exist_ok=True)
                df.to_csv(output_path, index=False, float_format='%.3f')
                logger.info(f"Detailed tracking data exported to {output_path}")
                return True
            else:
                logger.warning("No tracking data rows generated for export.")
                return False

        except Exception as e:
            logger.error(f"Error exporting tracking data: {str(e)}\n{traceback.format_exc()}")
            return False

    def get_performance_metrics(self):
        """Consolidated performance metrics."""
        total_time = time.time() - self.start_time
        avg_fps = self.processed_frames / total_time if total_time > 0 and self.processed_frames > 0 else 0
        avg_process_time = np.mean(self.processing_times) if self.processing_times else 0

        reid_metrics = self.reid.get_metrics()
        ocr_metrics = self.ocr.get_metrics()

        # Count players identified by jersey vs GID in highlight data
        jersey_player_count = sum(1 for k in self.player_tracking_data if isinstance(k, int))
        gid_player_count = sum(1 for k in self.player_tracking_data if isinstance(k, str) and k.startswith("gid_"))

        metrics = {
            'processed_frames': self.processed_frames,
            'total_processing_time_sec': total_time,
            'average_fps': avg_fps,
            'average_process_time_ms': avg_process_time * 1000,
            'current_frame': self.frame_idx,
            'reid_metrics': reid_metrics,
            'ocr_metrics': ocr_metrics,
            'active_tracks_last_frame': len(self.global_id_last_seen),
            'inactive_tracks_pool': len(self.inactive_tracks),
            'temporal_reappearances_matched': self._reappearances_matched,
            'total_tracked_players': len(self.player_tracking_data),
            'jersey_identified_players': jersey_player_count,
            'gid_identified_players': gid_player_count,
            'feature_clusters': len(self.reid.global_id_clusters) if hasattr(self.reid, 'global_id_clusters') else 0,
            'cluster_updates': self.reid.metrics.get('cluster_updates', 0),
            'temporal_buffer_matches': self.reid.metrics.get('temporal_matches', 0)
        }
        return metrics

# --- Standalone Highlight Extraction Function ---
def extract_player_highlights(video_path, highlight_data_path, output_dir="player_highlights", border_factor=0.1, selected_identifiers=None, max_clips=5, highlight_seconds=2.0, mode='merged'):
    """
    Extracts highlight video clips for players based on generated highlight data.
    Handles both jersey number (int) and GID string ("gid_X") identifiers.

    Args:
        video_path (str): Path to the original source video file.
        highlight_data_path (str): Path to the JSON file from generate_player_highlight_data.
        output_dir (str): Directory where highlight MP4 files will be saved.
        border_factor (float): Adds border around player bbox (0.1 = 10%). 0 for none.
        selected_identifiers (list, optional): List of jersey numbers (int) or GID strings ("gid_X") to process. None for all.
        max_clips (int): Maximum number of clips per player.
        highlight_seconds (float): Number of seconds to show the circle highlight at start.
        mode (str): 'merged' (default) for one video per player with all segments, 'partial' for separate clips per segment.
    """
    # Load Highlight Data
    try:
        with open(highlight_data_path, 'r') as f:
            highlight_data = json.load(f)
        if not highlight_data:
            logger.warning(f"Highlight data file {highlight_data_path} is empty.")
            return {}
    except FileNotFoundError:
        logger.error(f"Highlight data file not found: {highlight_data_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON {highlight_data_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading highlight data from {highlight_data_path}: {str(e)}\n{traceback.format_exc()}")
        return {}

    logger.info(f"Loaded highlight data for {len(highlight_data)} identifiers from {highlight_data_path}")

    # Prepare Video Source
    source_cap = cv2.VideoCapture(video_path)
    if not source_cap.isOpened():
        logger.error(f"Could not open source video: {video_path}")
        return {}
    fps = source_cap.get(cv2.CAP_PROP_FPS)
    w = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_vid = int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Source video: {w}x{h} @ {fps:.2f} FPS, {total_frames_vid} frames.")
    
    # Calculate highlight frames duration
    highlight_frames = int(highlight_seconds * fps)
    logger.info(f"Will highlight player with circle for first {highlight_frames} frames (~{highlight_seconds:.1f}s) of each segment")

    os.makedirs(output_dir, exist_ok=True)

    # Process Each Player/Identifier
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    font = cv2.FONT_HERSHEY_SIMPLEX
    extracted_clips_map = defaultdict(list)  # Store paths of created clips

    identifiers_to_process = selected_identifiers if selected_identifiers else list(highlight_data.keys())

    # Convert string keys from JSON back if they represent numbers
    processed_highlight_data = {}
    for k, v in highlight_data.items():
         try: processed_highlight_data[int(k)] = v  # Try converting key to int (jersey)
         except ValueError: processed_highlight_data[k] = v  # Keep as string (gid_X)

    for identifier in identifiers_to_process:
         # Ensure identifier is in the correct format (int or str) for lookup
         lookup_key = identifier
         if isinstance(identifier, int):
             lookup_key = identifier
         elif isinstance(identifier, str) and identifier.isdigit():
             lookup_key = int(identifier)  # Convert numeric strings from argparse
         elif isinstance(identifier, str) and not identifier.startswith("gid_"):
             logger.warning(f"Skipping invalid selected identifier: {identifier}. Must be jersey number or 'gid_X'.")
             continue

         # Get data for this identifier
         player_data = processed_highlight_data.get(lookup_key)
         if not player_data:
             logger.warning(f"No highlight data found for identifier '{lookup_key}', skipping.")
             continue

         # Determine label for video overlay (but won't be used in the actual video)
         player_label = f"Player #{lookup_key}" if isinstance(lookup_key, int) else f"Player {lookup_key}"
         output_prefix = f"player_{lookup_key}" if isinstance(lookup_key, int) else f"player_{lookup_key.replace('gid_','GID')}"

         segments = player_data.get('segments', [])
         if not segments: 
             logger.warning(f"No segments found for {player_label}, skipping.")
             continue

         # Sort segments by duration (longest first)
         segments.sort(key=lambda s: s.get('duration_sec', 0), reverse=True)
         segments_to_extract = segments[:max_clips]

         logger.info(f"Processing {len(segments_to_extract)} highlight segments for {player_label}...")

         if mode == 'merged':
             # Create a single merged video for all segments of this player
             output_filename = f"{output_prefix}_highlights.mp4"
             output_filepath = os.path.join(output_dir, output_filename)
             
             writer = None
             try:
                 writer = cv2.VideoWriter(output_filepath, fourcc, fps, (w, h))
                 if not writer.isOpened():
                     logger.error(f"Could not open video writer for {output_filepath}")
                     continue
                     
                 # Track total frames written for the merged video
                 total_frames_written = 0
                 
                 # Process each segment
                 for seg_idx, segment in enumerate(segments_to_extract):
                     start_frame = segment.get('start_frame')
                     end_frame = segment.get('end_frame')
                     track_coords = segment.get('track_coords', [])
                     
                     if start_frame is None or end_frame is None or start_frame > end_frame or not track_coords:
                         logger.warning(f"Skipping invalid segment {seg_idx+1} for {player_label}.")
                         continue
                     
                     # Validate frame range
                     if start_frame >= total_frames_vid:
                         logger.warning(f"Start frame {start_frame} is beyond video length {total_frames_vid}. Skipping segment.")
                         continue
                     
                     # Adjust end_frame if it exceeds video length
                     if end_frame >= total_frames_vid:
                         logger.warning(f"End frame {end_frame} exceeds video length {total_frames_vid}. Adjusting to video end.")
                         end_frame = total_frames_vid - 1
                     
                     # Create coordinate map for quick lookup
                     coord_map = {coord['frame']: coord for coord in track_coords if coord.get('bbox') is not None}
                     
                     # Improved seeking with fallback
                     seek_success = source_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                     if not seek_success:
                         logger.warning(f"Direct seek to frame {start_frame} failed. Using sequential read approach.")
                         # Reset to beginning and read frames until target
                         source_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                         for _ in range(start_frame):
                             success = source_cap.grab()  # Faster than read() as we don't need the frame data
                             if not success:
                                 break
                     
                     segment_frames_written = 0
                     current_frame = start_frame
                     
                     # Add max attempts to prevent infinite loop
                     max_frame_read_attempts = min(end_frame - start_frame + 10, 1000)  # reasonable limit
                     frame_read_attempts = 0
                     
                     # Process frames for this segment
                     while current_frame <= end_frame and frame_read_attempts < max_frame_read_attempts:
                         ret, frame = source_cap.read()
                         frame_read_attempts += 1
                         
                         if not ret:
                             logger.warning(f"Failed to read frame {current_frame} after {frame_read_attempts} attempts. "
                                           f"Video may be corrupt or end of file reached.")
                             break
                         
                         # Decide whether to highlight based on frame position within segment
                         is_highlight_period = segment_frames_written < highlight_frames
                         
                         frame_highlighted = frame.copy()
                         
                         # Only draw the player highlighting during the initial period
                         if is_highlight_period and current_frame in coord_map:
                             bbox = coord_map[current_frame]['bbox']
                             x1, y1, x2, y2 = map(int, bbox)
                             
                             # Calculate center and radius
                             center_x = (x1 + x2) // 2
                             center_y = (y1 + y2) // 2
                             radius = max((x2 - x1), (y2 - y1)) // 2
                             radius += int(radius * 0.2)  # Add 20% to radius for visibility
                             
                             # Draw circle around the player
                             cv2.circle(frame_highlighted, (center_x, center_y), radius, (0, 255, 255), 3, cv2.LINE_AA)
                         
                         writer.write(frame_highlighted)
                         segment_frames_written += 1
                         total_frames_written += 1
                         current_frame += 1
                     
                     logger.info(f"Added segment {seg_idx+1} to {player_label}'s highlight video: {segment_frames_written} frames")
                 
                 writer.release()
                 writer = None
                 
                 if total_frames_written > 0:
                     logger.info(f"Merged highlight video saved: {output_filepath} ({total_frames_written} frames)")
                     extracted_clips_map[lookup_key].append(output_filepath)
                 else:
                     logger.warning(f"No frames written for merged video: {output_filepath}")
                     if os.path.exists(output_filepath): os.remove(output_filepath)
                 
             except Exception as clip_err:
                 logger.error(f"Error creating merged highlight video {output_filename}: {str(clip_err)}\n{traceback.format_exc()}")
                 if writer is not None and writer.isOpened(): writer.release()
                 if os.path.exists(output_filepath): os.remove(output_filepath)
                 
         else:  # mode == 'partial' - create separate clips for each segment
             for seg_idx, segment in enumerate(segments_to_extract):
                 start_frame = segment.get('start_frame')
                 end_frame = segment.get('end_frame')
                 track_coords = segment.get('track_coords', [])

                 if start_frame is None or end_frame is None or start_frame > end_frame or not track_coords:
                     logger.warning(f"Skipping invalid segment {seg_idx+1} for {player_label}.")
                     continue
                     
                 # Validate frame range is within video bounds
                 if start_frame >= total_frames_vid:
                     logger.warning(f"Start frame {start_frame} is beyond video length {total_frames_vid}. Skipping segment.")
                     continue
                     
                 # Adjust end_frame if it exceeds video length
                 if end_frame >= total_frames_vid:
                     logger.warning(f"End frame {end_frame} exceeds video length {total_frames_vid}. Adjusting to video end.")
                     end_frame = total_frames_vid - 1

                 output_filename = f"{output_prefix}_clip_{seg_idx+1}.mp4"
                 output_filepath = os.path.join(output_dir, output_filename)

                 writer = None  # Initialize writer to None
                 try:
                     writer = cv2.VideoWriter(output_filepath, fourcc, fps, (w, h))
                     if not writer.isOpened():
                          logger.error(f"Could not open video writer for {output_filepath}")
                          continue

                     coord_map = {coord['frame']: coord for coord in track_coords if coord.get('bbox') is not None}

                     # Improved seeking with fallback
                     seek_success = source_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                     if not seek_success:
                         logger.warning(f"Direct seek to frame {start_frame} failed. Using sequential read approach.")
                         # Reset to beginning and read frames until target
                         source_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                         for _ in range(start_frame):
                             success = source_cap.grab()  # Faster than read() as we don't need the frame data
                             if not success:
                                 break

                     frames_written = 0
                     current_frame = start_frame
                     
                     # Add max attempts to prevent infinite loop
                     max_frame_read_attempts = min(end_frame - start_frame + 10, 1000)  # reasonable limit
                     frame_read_attempts = 0
                     
                     while current_frame <= end_frame and frame_read_attempts < max_frame_read_attempts:
                         ret, frame = source_cap.read()
                         frame_read_attempts += 1
                         
                         if not ret:
                             logger.warning(f"Failed to read frame {current_frame} after {frame_read_attempts} attempts. "
                                          f"Video may be corrupt or end of file reached.")
                             break

                         # Decide whether to highlight based on frame position
                         is_highlight_period = frames_written < highlight_frames
                         
                         frame_highlighted = frame.copy()

                         # Only draw the player highlighting during the initial period
                         if is_highlight_period and current_frame in coord_map:
                             bbox = coord_map[current_frame]['bbox']
                             x1, y1, x2, y2 = map(int, bbox)
                             
                             # Calculate center and radius
                             center_x = (x1 + x2) // 2
                             center_y = (y1 + y2) // 2
                             radius = max((x2 - x1), (y2 - y1)) // 2
                             radius += int(radius * 0.2)  # Add 20% to radius for visibility
                             
                             # Draw circle around the player
                             cv2.circle(frame_highlighted, (center_x, center_y), radius, (0, 255, 255), 3, cv2.LINE_AA)
                         
                         writer.write(frame_highlighted)
                         frames_written += 1
                         current_frame += 1

                     writer.release()
                     writer = None  # Reset writer after release
                     if frames_written > 0:
                         logger.info(f"Highlight clip saved: {output_filepath} ({frames_written} frames)")
                         extracted_clips_map[lookup_key].append(output_filepath)
                     else:
                         logger.warning(f"No frames written for clip: {output_filepath}")
                         if os.path.exists(output_filepath): os.remove(output_filepath)

                 except Exception as clip_err:
                     logger.error(f"Error creating highlight clip {output_filename}: {str(clip_err)}\n{traceback.format_exc()}")
                     if writer is not None and writer.isOpened(): writer.release()  # Ensure release on error
                     if os.path.exists(output_filepath): os.remove(output_filepath)  # Clean up partial file

    source_cap.release()
    logger.info("Finished highlight extraction process.")
    return dict(extracted_clips_map)  # Convert back to dict


# --------------------- Main Execution --------------------- #
def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser("Enhanced Soccer Player Tracker")
    # Input/Output
    parser.add_argument("--video", default='videos/00005(1).mov', help="Path to input video")
    parser.add_argument("--output-dir", default="output", help="Directory for all output files")
    parser.add_argument("--output-video", default="tracked_video.mp4", help="Filename for the annotated output video (relative to output-dir)")
    # Models
    parser.add_argument("--yolo-model", default="yolo11x.pt", help="YOLO model path")
    parser.add_argument("--reid-model", default="osnet_ain_x1_0", help="TorchReID model name")
    parser.add_argument("--reid-weights", default="osnet_ain_x1_0.pth", help="ReID model weights path")
    parser.add_argument("--tracker-cfg", default="custom.yaml", help="Ultralytics tracker config file path (e.g., bytetrack.yaml, botsort.yaml)")
    # Tracking Parameters
    parser.add_argument("--conf-thres", type=float, default=0.3, help="YOLO detection confidence threshold")
    parser.add_argument("--reid-threshold", type=float, default=0.8, help="ReID similarity threshold for matching")
    parser.add_argument("--iou-match-threshold", type=float, default=0.3, help="Min IoU for OCR-player box association")
    parser.add_argument("--ocr-interval", type=int, default=5, help="Run OCR every N frames (0 to disable)")
    parser.add_argument("--temporal-window", type=int, default=100, help="Max frames (time window) for inactive track matching")
    parser.add_argument("--max-history", type=int, default=100, help="Max frames for feature/trajectory history storage")
    parser.add_argument("--motion-weight", type=float, default=0.3, help="Weight for motion prediction in inactive matching (0-1)")
    # Feature clustering params
    parser.add_argument("--use-clustering", type=bool, default=True, help="Use feature clustering for better ReID matching")
    parser.add_argument("--num-clusters", type=int, default=3, help="Number of feature clusters per player identity")
    parser.add_argument("--reid-weight", type=float, default=0.7, help="Weight for ReID score in fusion formula (0-1)")
    parser.add_argument("--jersey-weight", type=float, default=0.3, help="Weight for jersey score in fusion formula (0-1)")
    # Frame Control
    parser.add_argument("--fps", type=float, default=0, help="Override video FPS (0 to use source FPS)")
    parser.add_argument("--skip-frames", type=int, default=0, help="Process every N+1 frame (e.g., 0=all, 1=every 2nd)")
    parser.add_argument("--start-frame", type=int, default=0, help="Start processing from this frame index")
    parser.add_argument("--end-frame", type=int, default=-1, help="Stop processing at this frame index (-1 for end)")
    # Features & Outputs
    parser.add_argument("--display", action="store_true", help="Show tracking window (can slow down processing)")
    parser.add_argument("--show-trails", action="store_true", help="Draw motion trails on output video")
    parser.add_argument("--export-csv", action="store_true", help="Export detailed tracking data to CSV")
    parser.add_argument("--generate-trajectory", action="store_true", help="Generate trajectory visualization image")
    parser.add_argument("--export-trajectories", action="store_true", help="Export raw trajectory data for all players to JSON")
    parser.add_argument("--extract-highlights", action="store_true", help="Generate highlight data (JSON) and extract video clips")
    parser.add_argument("--highlight-players", type=str, help="Comma-separated list of identifiers (jersey numbers or 'gid_X') to extract highlights/data for (default: all)")
    parser.add_argument("--max-clips", type=int, default=10, help="Maximum highlight clips per player")
    parser.add_argument("--highlight-seconds", type=float, default=1.5, help="Number of seconds to highlight player with circle at clip start")
    # Misc
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Computation device")

    args = parser.parse_args()

    # --- Validate Paths ---
    if not os.path.isfile(args.video): logger.error(f"Video not found: {args.video}"); sys.exit(1)
    if not os.path.isfile(args.yolo_model): logger.error(f"YOLO model not found: {args.yolo_model}"); sys.exit(1)
    if not os.path.isfile(args.reid_weights): logger.error(f"ReID weights not found: {args.reid_weights}"); sys.exit(1)
    if not os.path.isfile(args.tracker_cfg): logger.error(f"Tracker config not found: {args.tracker_cfg}"); sys.exit(1)

    # --- Create Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # --- Parse Selected Players/Identifiers ---
    selected_identifiers_list = None
    if args.highlight_players:
        try:
            selected_identifiers_list = []
            for item in args.highlight_players.split(','):
                item = item.strip()
                if item.isdigit():
                    selected_identifiers_list.append(int(item))
                elif item.lower().startswith("gid_") and item[4:].isdigit():
                     selected_identifiers_list.append(item.lower()) # Keep as "gid_X" string
                elif item: # Only warn if item is not empty
                    logger.warning(f"Ignoring invalid identifier in --highlight-players: '{item}'")
            if not selected_identifiers_list:
                 logger.warning("--highlight-players specified but no valid identifiers found.")
            else:
                 logger.info(f"Processing will focus on identifiers: {selected_identifiers_list}")
        except Exception as e:
            logger.error(f"Error parsing --highlight-players: {e}")
            sys.exit(1)

    # --- Get Video Properties ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): logger.error(f"Cannot open video: {args.video}"); sys.exit(1)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release() # Release now, will reopen for processing

    fps = args.fps if args.fps > 0 else source_fps
    if fps <= 0: fps = 30.0; logger.warning("Invalid source FPS, using default 30.0.")
    logger.info(f"Video properties: {w}x{h} @ {fps:.2f} FPS (effective), {total_frames_vid} total frames.")

    # --- Initialize Tracker ---
    try:
        tracker = SoccerTracker(
            yolo_model_path=args.yolo_model, reid_model_name=args.reid_model,
            reid_weights_path=args.reid_weights, conf=args.conf_thres,
            reid_threshold=args.reid_threshold, tracker_cfg=args.tracker_cfg,
            device=args.device, ocr_interval=args.ocr_interval,
            max_history=args.max_history, temporal_window=args.temporal_window,
            iou_match_threshold=args.iou_match_threshold, fps=fps,
            motion_prediction_weight=args.motion_weight,
            use_clustering=args.use_clustering,
            reid_weight=args.reid_weight,
            jersey_weight=args.jersey_weight
        )
    except Exception as e:
        logger.error(f"Fatal Error initializing tracker: {str(e)}\n{traceback.format_exc()}"); sys.exit(1)

    # --- Prepare Video I/O ---
    cap = cv2.VideoCapture(args.video) # Reopen for processing
    if not cap.isOpened(): logger.error(f"Cannot reopen video: {args.video}"); sys.exit(1)

    output_video_path = os.path.join(args.output_dir, args.output_video)
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        if not writer.isOpened(): raise IOError(f"Could not open writer for {output_video_path}")
    except Exception as e:
        logger.error(f"Fatal error creating output video writer: {e}"); cap.release(); sys.exit(1)

    # --- Frame Processing Loop ---
    start_frame = max(0, args.start_frame)
    end_frame = total_frames_vid if args.end_frame <= 0 else min(total_frames_vid, args.end_frame)
    frame_skip_interval = args.skip_frames + 1 # Process frame 0, frame N+1, frame 2N+2, ...

    if start_frame >= end_frame:
         logger.error(f"Start frame ({start_frame}) is greater than or equal to end frame ({end_frame}). No frames to process.")
         cap.release(); writer.release(); sys.exit(1)

    if start_frame > 0:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
             logger.warning(f"Could not accurately seek to start frame {start_frame}. Processing may start earlier.")
        else:
             logger.info(f"Seeked to start frame {start_frame}.")

    frames_to_process_estimate = (end_frame - start_frame + frame_skip_interval - 1) // frame_skip_interval
    pbar = tqdm(total=frames_to_process_estimate, desc="Tracking Progress", unit="frame")
    current_frame_num = start_frame

    try:
        while current_frame_num < end_frame:
            # Read frame first
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {current_frame_num}, stopping.")
                break

            # Decide whether to process this frame based on skip interval
            if (current_frame_num - start_frame) % frame_skip_interval == 0:
                # Process the frame
                try:
                    # Pass the *original* frame number to process_frame
                    tracker.frame_idx = current_frame_num # Set tracker's internal frame index
                    annotated_frame, _ = tracker.process_frame(frame) # Ignore tracks_data return value here
                    writer.write(annotated_frame)

                    if args.display:
                        display_frame = cv2.resize(annotated_frame, (1280, 720)) if w > 1920 else annotated_frame
                        cv2.imshow("Soccer Tracking", display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'): logger.info("Stop requested by user."); break

                    pbar.update(1)

                    # Log stats periodically
                    if pbar.n % 100 == 0 or pbar.n == frames_to_process_estimate:
                        metrics = tracker.get_performance_metrics()
                        logger.info(
                            f"Frame {current_frame_num}/{total_frames_vid} (Processed {pbar.n}/{frames_to_process_estimate}) | "
                            f"Avg FPS: {metrics.get('average_fps',0):.2f} | "
                            f"Tracked Players: {metrics.get('total_tracked_players',0)} ({metrics.get('jersey_identified_players',0)} Jersey)"
                        )

                except Exception as e:
                    logger.error(f"Error processing frame {current_frame_num}: {str(e)}\n{traceback.format_exc()}")
                    writer.write(frame) # Write original frame on error

            # Move to the next frame in the video
            current_frame_num += 1

            # If skipping frames, advance the video capture position quickly
            # Note: This might not be perfectly accurate but avoids reading every frame
            if frame_skip_interval > 1 and (current_frame_num - start_frame) % frame_skip_interval != 0:
                 # Try setting position directly (less reliable)
                 # success = cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
                 # Or just continue reading until the next desired frame (more reliable but slower if skipping many)
                 pass # Let the loop read the next frame

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user (Ctrl+C).")
    finally:
        pbar.close()
        cap.release()
        writer.release()
        if args.display: cv2.destroyAllWindows()
        logger.info("Video capture and writer released.")

    # --- Final Output Generation ---
    try:
        logger.info("Generating final outputs...")
        metrics_file_path = os.path.join(args.output_dir, "performance_metrics.json")
        highlight_data_path = os.path.join(args.output_dir, "highlight_data.json")
        csv_path = os.path.join(args.output_dir, "detailed_tracking_data.csv")
        trajectory_path = os.path.join(args.output_dir, "player_trajectories.jpg")
        trajectories_json_path = os.path.join(args.output_dir, "player_raw_trajectories.json")

        # 1. Performance Metrics
        final_metrics = tracker.get_performance_metrics()
        try:
             with open(metrics_file_path, 'w') as f: json.dump(final_metrics, f, indent=2)
             logger.info(f"Performance metrics saved to: {metrics_file_path}")
        except Exception as e: logger.error(f"Failed to save metrics: {e}")

        # Log key metrics
        logger.info("--- Tracking Complete. Final Performance ---")
        logger.info(f"Processed {final_metrics['processed_frames']} frames in {final_metrics['total_processing_time_sec']:.2f} seconds")
        logger.info(f"Average FPS: {final_metrics['average_fps']:.2f}")
        logger.info(f"Avg Proc Time/Frame: {final_metrics['average_process_time_ms']:.2f} ms")
        logger.info(f"Total GIDs Created: {final_metrics['reid_metrics']['total_global_ids_ever']}")
        logger.info(f"Final Active GIDs: {final_metrics['reid_metrics']['active_global_ids']}")
        logger.info(f"Temporal/Reappearance Matches: {final_metrics['temporal_reappearances_matched']}")
        logger.info(f"Feature Clusters: {final_metrics['feature_clusters']}")
        logger.info(f"Total Players Tracked (Highlights): {final_metrics['total_tracked_players']}")
        logger.info(f"  - By Jersey: {final_metrics['jersey_identified_players']}")
        logger.info(f"  - By GID: {final_metrics['gid_identified_players']}")
        logger.info("-------------------------------------------")

        # 2. Generate Trajectory Visualization (Optional) - Include ALL players
        if args.generate_trajectory:
            logger.info("Generating trajectory visualization for ALL players...")
            # Pass None as selected_players to include all trajectories
            tracker.generate_trajectory(trajectory_path, frame_size=(w, h), selected_players=None)
            
        # 2b. Export raw trajectory data for all players (New feature)
        if args.export_trajectories:
            logger.info("Exporting raw trajectory data for ALL players...")
            try:
                # Convert track_history_viz to serializable format
                trajectory_export = {}
                for gid, history in tracker.track_history_viz.items():
                    jersey_num, conf = tracker.reid.get_best_jersey(gid)
                    history_list = [{
                        'frame': entry.get('frame'),
                        'x': float(entry.get('x', 0)),
                        'y': float(entry.get('y', 0)),
                        'bbox': [float(v) for v in entry.get('bbox', [0,0,0,0])]
                    } for entry in list(history) if entry.get('x') is not None and entry.get('y') is not None]
                    
                    # Add player identifier in the data
                    trajectory_export[f"gid_{gid}"] = {
                        'jersey_number': jersey_num,
                        'jersey_confidence': float(conf),
                        'trajectory': history_list
                    }
                
                # Save as JSON with special encoder for numpy types
                class NpEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer): return int(obj)
                        if isinstance(obj, np.floating): return float(obj)
                        if isinstance(obj, np.ndarray): return obj.tolist()
                        return super(NpEncoder, self).default(obj)
                        
                with open(trajectories_json_path, 'w') as f:
                    json.dump(trajectory_export, f, cls=NpEncoder, indent=2)
                logger.info(f"Raw trajectory data for ALL players exported to: {trajectories_json_path}")
            except Exception as e:
                logger.error(f"Error exporting raw trajectories: {str(e)}\n{traceback.format_exc()}")

        # 3. Export Detailed Tracking Data (Optional) - Include ALL players
        if args.export_csv:
            logger.info("Exporting detailed tracking data for ALL players...")
            # Pass None as selected_identifiers to include ALL players
            tracker.export_tracking_data(csv_path, selected_identifiers=None)

        # 4. Generate Highlight Data JSON (Required if extracting clips)
        highlight_data = None
        if args.extract_highlights or not os.path.exists(highlight_data_path): # Generate if extracting or not present
             logger.info("Generating highlight data JSON for ALL players...")
             highlight_data = tracker.generate_player_highlight_data(
                 output_path=highlight_data_path,
                 min_segment_duration=2.0 # Segments must be at least 2s
             )
        elif args.extract_highlights: # Load existing if extracting and file exists
             try:
                 with open(highlight_data_path, 'r') as f: highlight_data = json.load(f)
                 logger.info(f"Loaded existing highlight data from {highlight_data_path}")
             except Exception as e:
                 logger.error(f"Failed to load existing highlight data {highlight_data_path}: {e}. Cannot extract clips.")
                 highlight_data = None

        # 5. Extract Highlight Video Clips (Optional)
        if args.extract_highlights:
            if highlight_data:
                logger.info("Starting extraction of highlight video clips...")
                highlight_clip_dir = os.path.join(args.output_dir, "highlight_clips")
                extract_player_highlights(
                    video_path=args.video,
                    highlight_data_path=highlight_data_path, # Use the JSON path
                    output_dir=highlight_clip_dir,
                    selected_identifiers=selected_identifiers_list,  # Will use ALL if None
                    max_clips=args.max_clips,
                    highlight_seconds=args.highlight_seconds
                )
                logger.info(f"Highlight video extraction process finished. Check directory: {highlight_clip_dir}")
            else:
                logger.warning("Skipping highlight video extraction: Highlight data is missing or failed to generate/load.")

        logger.info(f"All outputs saved in directory: {args.output_dir}")

    except Exception as e:
        logger.error(f"Error during final output generation: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()