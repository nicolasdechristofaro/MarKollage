#!/usr/bin/env python3
"""
MarkovPhotoCropper: A Markov chain-driven photo sequence generator

This system analyzes photos using advanced composition principles to detect interesting regions,
generates meaningful crops, and uses a Markov chain to create engaging photo sequences.
"""

import cv2
import numpy as np
from skimage import feature
import random
import os
from typing import List, Tuple, Dict, Optional
import json
from scipy import ndimage
from scipy.spatial.distance import cdist


class PhotoAnalyzer:
    """Analyzes photos using advanced composition principles to detect interesting regions."""
    
    def __init__(self):
        print("🎨 Advanced composition analyzer loaded")
    
    def detect_visual_elements(self, image: np.ndarray) -> List[Dict]:
        """Detect visual elements using advanced composition analysis."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        visual_elements = []
        
        # 1. Detect strong edges and lines (potential leading lines)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                # Score lines based on length and angle (diagonal lines are more interesting)
                line_score = min(1.0, length / 200) * (1.0 - abs(angle % 90) / 90)
                
                if line_score > 0.3:
                    visual_elements.append({
                        'type': 'leading_line',
                        'bbox': (min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)),
                        'score': line_score,
                        'center': ((x1+x2)//2, (y1+y2)//2),
                        'angle': angle,
                        'length': length
                    })
        
        # 2. Detect corners and interest points (potential focal points)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=50)
        
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel().astype(int)
                # Create a small region around the corner
                size = 100
                x1, y1 = max(0, x-size//2), max(0, y-size//2)
                x2, y2 = min(width, x+size//2), min(height, y+size//2)
                
                visual_elements.append({
                    'type': 'focal_point',
                    'bbox': (x1, y1, x2-x1, y2-y1),
                    'score': 0.7,  # High score for focal points
                    'center': (x, y),
                    'corner_strength': corner[0][2] if len(corner[0]) > 2 else 0.5
                })
        
        # 3. Detect high contrast regions (potential subjects)
        # Use adaptive thresholding to find high contrast areas
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Only significant regions
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate contrast score
                roi = gray[y:y+h, x:x+w]
                contrast = np.std(roi) / 255.0
                
                if contrast > 0.3:
                    visual_elements.append({
                        'type': 'high_contrast',
                        'bbox': (x, y, w, h),
                        'score': contrast,
                        'center': (x + w//2, y + h//2),
                        'area': area,
                        'contrast': contrast
                    })
        
        return visual_elements
    
    def find_interesting_regions(self, image: np.ndarray, min_size: int = 100) -> List[Dict]:
        """Find interesting regions using advanced composition analysis."""
        visual_elements = self.detect_visual_elements(image)
        
        # Filter and enhance visual elements
        interesting_regions = []
        
        for element in visual_elements:
            x, y, w, h = element['bbox']
            
            # Skip elements that are too small
            if w < min_size or h < min_size:
                continue
            
            # Enhance score based on element type and properties
            base_score = element['score']
            
            # Additional scoring based on element type
            if element['type'] == 'leading_line':
                # Longer, more diagonal lines get higher scores
                length_bonus = min(0.3, element.get('length', 0) / 500)
                angle_bonus = 0.2 if 20 < abs(element.get('angle', 0) % 90) < 70 else 0.1
                base_score += length_bonus + angle_bonus
                
            elif element['type'] == 'focal_point':
                # Focal points get consistent high scores
                base_score = max(0.7, base_score)
                
            elif element['type'] == 'high_contrast':
                # High contrast regions get bonus for size and contrast
                size_bonus = min(0.2, element.get('area', 0) / 10000)
                contrast_bonus = element.get('contrast', 0) * 0.3
                base_score += size_bonus + contrast_bonus
            
            # Final score
            final_score = min(0.95, base_score)
            
            interesting_regions.append({
                'type': element['type'],
                'bbox': (x, y, w, h),
                'score': final_score,
                'center': element['center'],
                'element_data': element
            })
        
        # Sort by score and return top regions
        interesting_regions.sort(key=lambda x: x['score'], reverse=True)
        
        return interesting_regions[:30]  # Return top 30 regions
    


class CropGenerator:
    """Generates meaningful crops from detected regions with photography composition principles."""
    
    def __init__(self, target_aspect_ratios: List[Tuple[int, int]] = None):
        # Use only landscape aspect ratios for uniformity
        self.target_aspect_ratios = target_aspect_ratios or [
            (16, 9),   # Widescreen landscape
            (4, 3),    # Traditional landscape
            (3, 2),    # 35mm landscape
        ]
    
    def analyze_composition(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """Analyze advanced photography composition principles for a crop."""
        x, y, w, h = bbox
        crop_image = image[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        crop_height, crop_width = gray.shape
        
        # 1. Rule of Thirds Analysis
        rule_of_thirds_score = self._analyze_rule_of_thirds(gray)
        
        # 2. Visual Hierarchy Analysis
        hierarchy_score = self._analyze_visual_hierarchy(gray, crop_image)
        
        # 3. Leading Lines Analysis
        leading_lines_score = self._analyze_leading_lines(gray)
        
        # 4. Symmetry and Balance Analysis
        symmetry_score = self._analyze_symmetry_balance(gray)
        
        # 5. Color Harmony Analysis
        color_harmony_score = self._analyze_color_harmony(crop_image)
        
        # 6. Depth and Layering Analysis
        depth_score = self._analyze_depth_layering(gray)
        
        # 7. Negative Space Analysis
        negative_space_score = self._analyze_negative_space(gray)
        
        # 8. Geometric Composition Analysis
        geometric_score = self._analyze_geometric_composition(gray)
        
        # 9. Contrast and Tonal Range Analysis
        contrast_score = self._analyze_contrast_tonal(gray)
        
        # 10. Focal Point Analysis
        focal_point_score = self._analyze_focal_points(gray)
        
        # Weighted composition score with emphasis on key principles
        composition_score = (
            0.15 * rule_of_thirds_score +      # Classic composition
            0.12 * hierarchy_score +           # Visual flow
            0.12 * leading_lines_score +       # Direction and movement
            0.10 * symmetry_score +            # Balance and harmony
            0.10 * color_harmony_score +       # Color relationships
            0.10 * depth_score +               # Depth perception
            0.08 * negative_space_score +      # Breathing room
            0.08 * geometric_score +           # Shape relationships
            0.08 * contrast_score +            # Visual impact
            0.07 * focal_point_score           # Subject emphasis
        )
        
        return {
            'rule_of_thirds': rule_of_thirds_score,
            'visual_hierarchy': hierarchy_score,
            'leading_lines': leading_lines_score,
            'symmetry_balance': symmetry_score,
            'color_harmony': color_harmony_score,
            'depth_layering': depth_score,
            'negative_space': negative_space_score,
            'geometric_composition': geometric_score,
            'contrast_tonal': contrast_score,
            'focal_points': focal_point_score,
            'composition_score': composition_score
        }
    
    
    def generate_crops(self, image: np.ndarray, regions: List[Dict], 
                      num_crops: int = 15) -> List[Dict]:
        """Generate diverse, well-composed crops using photography principles."""
        height, width = image.shape[:2]
        crops = []
        
        # Define larger landscape crop size (more zoomed out)
        crop_w = min(1600, width)  # Larger width for more zoomed out view
        crop_h = int(crop_w * 9 / 16)  # 16:9 aspect ratio
        
        # Generate a grid of potential crop positions across the entire image
        grid_positions = self._generate_grid_positions(width, height, crop_w, crop_h)
        
        
        for i, (crop_x, crop_y) in enumerate(grid_positions):
            # Analyze composition for this crop position
            composition = self.analyze_composition(image, (crop_x, crop_y, crop_w, crop_h))
            
            # Check if this crop contains any interesting visual elements
            element_presence = self._check_visual_element_presence(regions, (crop_x, crop_y, crop_w, crop_h))
            
            # Score based primarily on composition, with visual element presence as small bonus
            composition_score = composition['composition_score']
            element_bonus = element_presence * 0.1  # Very small bonus for having visual elements
            final_score = composition_score + element_bonus
            
            # Only keep crops with good composition
            if composition_score > 0.4:  # Minimum composition threshold
                crops.append({
                    'bbox': (crop_x, crop_y, crop_w, crop_h),
                    'aspect_ratio': (16, 9),
                    'score': final_score,
                    'region_type': 'composition',
                    'region_center': (crop_x + crop_w//2, crop_y + crop_h//2),
                    'coverage': element_presence,
                    'composition': composition
                })
        
        # Sort by composition score and remove overlapping crops
        crops.sort(key=lambda x: x['score'], reverse=True)
        diverse_crops = self._remove_overlapping_crops(crops)
        
        return diverse_crops[:num_crops]
    
    
    def _analyze_rule_of_thirds(self, gray: np.ndarray) -> float:
        """Advanced rule of thirds analysis with intersection strength."""
        height, width = gray.shape
        third_w, third_h = width // 3, height // 3
        
        # Calculate edge strength at rule of thirds intersections
        intersections = [
            (third_w, third_h), (2*third_w, third_h),
            (third_w, 2*third_h), (2*third_w, 2*third_h)
        ]
        
        intersection_strength = 0
        for v, h in intersections:
            if 0 <= v < width and 0 <= h < height:
                # Sample a region around the intersection
                region_size = min(20, width//10, height//10)
                y1, y2 = max(0, h-region_size), min(height, h+region_size)
                x1, x2 = max(0, v-region_size), min(width, v+region_size)
                region = gray[y1:y2, x1:x2]
                
                if region.size > 0:
                    # Calculate local contrast and edge strength
                    local_contrast = np.std(region)
                    edges = cv2.Canny(region, 50, 150)
                    edge_density = np.sum(edges > 0) / region.size
                    intersection_strength += local_contrast * (1 + edge_density)
        
        # Also check for strong elements along the rule of thirds lines
        line_strength = 0
        for v in [third_w, 2*third_w]:
            if 0 <= v < width:
                line_region = gray[:, max(0, v-5):min(width, v+5)]
                line_strength += np.std(line_region)
        
        for h in [third_h, 2*third_h]:
            if 0 <= h < height:
                line_region = gray[max(0, h-5):min(height, h+5), :]
                line_strength += np.std(line_region)
        
        # Combine intersection and line strength
        total_strength = intersection_strength + line_strength * 0.5
        return min(1.0, total_strength / 2000)
    
    def _quick_color_harmony(self, image: np.ndarray) -> float:
        """Quick color harmony analysis."""
        # Convert to HSV and analyze saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        
        # Good composition has balanced saturation and contrast
        saturation_score = np.mean(s) / 255.0
        contrast_score = np.std(v) / 255.0
        
        return min(1.0, (saturation_score + contrast_score) / 2)
    
    def _quick_depth(self, gray: np.ndarray) -> float:
        """Quick depth analysis using gradient variance."""
        # Simple gradient analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return min(1.0, np.std(gradient_magnitude) / 255.0 * 2)
    
    def _analyze_visual_hierarchy(self, gray: np.ndarray, color_image: np.ndarray) -> float:
        """Analyze visual hierarchy and focal points."""
        height, width = gray.shape
        
        # Find the brightest and darkest regions (potential focal points)
        bright_regions = gray > np.percentile(gray, 85)
        dark_regions = gray < np.percentile(gray, 15)
        
        # Calculate contrast between bright and dark regions
        if np.any(bright_regions) and np.any(dark_regions):
            bright_mean = np.mean(gray[bright_regions])
            dark_mean = np.mean(gray[dark_regions])
            contrast_ratio = (bright_mean - dark_mean) / 255.0
        else:
            contrast_ratio = 0
        
        # Analyze edge density (more edges = more visual interest)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Check for central focal point
        center_y, center_x = height // 2, width // 2
        center_region_size = min(50, width//4, height//4)
        center_region = gray[center_y-center_region_size:center_y+center_region_size,
                           center_x-center_region_size:center_x+center_region_size]
        
        center_contrast = np.std(center_region) if center_region.size > 0 else 0
        
        # Combine metrics
        hierarchy_score = (
            0.4 * min(1.0, contrast_ratio * 2) +
            0.3 * min(1.0, edge_density * 10) +
            0.3 * min(1.0, center_contrast / 50)
        )
        
        return hierarchy_score
    
    def _analyze_leading_lines(self, gray: np.ndarray) -> float:
        """Analyze leading lines and directional flow."""
        # Detect lines using Hough transform
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=50, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        # Analyze line directions and lengths
        line_scores = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            
            # Score lines based on length and direction
            length_score = min(1.0, length / 200)
            
            # Diagonal lines are more interesting than horizontal/vertical
            angle_score = 1.0 - abs(abs(angle % 90) - 45) / 45
            
            # Lines pointing toward center are more valuable
            center_x, center_y = gray.shape[1]//2, gray.shape[0]//2
            line_center_x, line_center_y = (x1+x2)/2, (y1+y2)/2
            distance_to_center = np.sqrt((line_center_x - center_x)**2 + (line_center_y - center_y)**2)
            center_score = 1.0 - min(1.0, distance_to_center / (gray.shape[0] + gray.shape[1])/2)
            
            line_score = length_score * (0.4 + 0.3 * angle_score + 0.3 * center_score)
            line_scores.append(line_score)
        
        # Return average score of all lines
        return min(1.0, np.mean(line_scores) if line_scores else 0.0)
    
    def _analyze_symmetry_balance(self, gray: np.ndarray) -> float:
        """Analyze symmetry and visual balance."""
        height, width = gray.shape
        
        # Check horizontal symmetry
        top_half = gray[:height//2, :]
        bottom_half = gray[height//2:, :]
        if top_half.shape != bottom_half.shape:
            bottom_half = bottom_half[:top_half.shape[0], :]
        
        horizontal_symmetry = 1.0 - np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float))) / 255.0
        
        # Check vertical symmetry
        left_half = gray[:, :width//2]
        right_half = gray[:, width//2:]
        if left_half.shape != right_half.shape:
            right_half = right_half[:, :left_half.shape[1]]
        
        vertical_symmetry = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
        
        # Balance analysis - check if visual weight is distributed
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        total_mass = np.sum(gray)
        if total_mass > 0:
            center_x = np.sum(x_coords * gray) / total_mass
            center_y = np.sum(y_coords * gray) / total_mass
            
            # Distance from geometric center indicates imbalance
            geometric_center_x, geometric_center_y = width/2, height/2
            balance_score = 1.0 - min(1.0, np.sqrt((center_x - geometric_center_x)**2 + (center_y - geometric_center_y)**2) / (width + height)/2)
        else:
            balance_score = 0.0
        
        # Combine symmetry and balance
        symmetry_score = (
            0.4 * horizontal_symmetry +
            0.4 * vertical_symmetry +
            0.2 * balance_score
        )
        
        return symmetry_score
    
    def _analyze_color_harmony(self, image: np.ndarray) -> float:
        """Analyze color harmony and relationships."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Calculate color distribution
        hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
        hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
        
        # Check for color concentration (not too scattered)
        dominant_h = np.argmax(hist_h)
        dominant_s = np.argmax(hist_s)
        dominant_v = np.argmax(hist_v)
        
        h_concentration = hist_h[dominant_h] / np.sum(hist_h)
        s_concentration = hist_s[dominant_s] / np.sum(hist_s)
        v_concentration = hist_v[dominant_v] / np.sum(hist_v)
        
        # Check for complementary colors (simplified)
        complementary_score = 0.0
        if len(hist_h) > 0:
            # Find second most dominant hue
            hist_h_sorted = np.argsort(hist_h.flatten())[::-1]
            if len(hist_h_sorted) > 1:
                dominant_hue = hist_h_sorted[0]
                second_hue = hist_h_sorted[1]
                hue_diff = abs(dominant_hue - second_hue)
                # Complementary colors are ~90 degrees apart
                if 80 <= hue_diff <= 100:
                    complementary_score = 0.3
        
        # Saturation and value harmony
        saturation_score = min(1.0, np.mean(s) / 128)  # Moderate saturation is good
        value_score = 1.0 - abs(np.mean(v) - 128) / 128  # Balanced brightness is good
        
        harmony_score = (
            0.3 * (h_concentration + s_concentration + v_concentration) / 3 +
            0.2 * complementary_score +
            0.25 * saturation_score +
            0.25 * value_score
        )
        
        return harmony_score
    
    def _analyze_depth_layering(self, gray: np.ndarray) -> float:
        """Analyze depth perception and layering."""
        # Calculate gradient magnitude for depth cues
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Higher gradient variation suggests more depth
        gradient_variation = np.std(gradient_magnitude)
        
        # Analyze focus/blur patterns (simplified)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Check for atmospheric perspective (darker in distance)
        height, width = gray.shape
        top_region = gray[:height//3, :]
        bottom_region = gray[2*height//3:, :]
        
        if top_region.size > 0 and bottom_region.size > 0:
            atmospheric_perspective = (np.mean(bottom_region) - np.mean(top_region)) / 255.0
        else:
            atmospheric_perspective = 0.0
        
        # Combine depth indicators
        depth_score = (
            0.4 * min(1.0, gradient_variation / 100) +
            0.3 * min(1.0, edge_density * 10) +
            0.3 * max(0.0, atmospheric_perspective)
        )
        
        return depth_score
    
    def _analyze_negative_space(self, gray: np.ndarray) -> float:
        """Analyze negative space and breathing room."""
        height, width = gray.shape
        
        # Find areas with low detail (potential negative space)
        edges = cv2.Canny(gray, 50, 150)
        
        # Divide image into regions and analyze edge density
        region_size = min(50, width//8, height//8)
        regions_x = width // region_size
        regions_y = height // region_size
        
        edge_densities = []
        for i in range(regions_y):
            for j in range(regions_x):
                y1, y2 = i * region_size, min((i+1) * region_size, height)
                x1, x2 = j * region_size, min((j+1) * region_size, width)
                
                region_edges = edges[y1:y2, x1:x2]
                edge_density = np.sum(region_edges > 0) / region_edges.size
                edge_densities.append(edge_density)
        
        if not edge_densities:
            return 0.5
        
        # Good negative space has a mix of busy and quiet areas
        edge_densities = np.array(edge_densities)
        density_variation = np.std(edge_densities)
        
        # Some quiet areas (low edge density) are good
        quiet_areas = np.sum(edge_densities < 0.1) / len(edge_densities)
        
        negative_space_score = (
            0.6 * min(1.0, density_variation * 5) +
            0.4 * min(1.0, quiet_areas * 2)
        )
        
        return negative_space_score
    
    def _analyze_geometric_composition(self, gray: np.ndarray) -> float:
        """Analyze geometric shapes and patterns."""
        # Detect geometric shapes using contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        geometric_scores = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip very small contours
                continue
            
            # Approximate the contour to detect shapes
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Score based on shape complexity and regularity
            vertices = len(approx)
            
            if vertices == 3:  # Triangle
                shape_score = 0.8
            elif vertices == 4:  # Rectangle/square
                shape_score = 0.7
            elif vertices > 8:  # Circle-like
                shape_score = 0.9
            else:  # Other polygons
                shape_score = 0.6
            
            # Bonus for larger shapes
            size_bonus = min(0.3, area / 10000)
            
            geometric_scores.append(shape_score + size_bonus)
        
        # Return average geometric score
        return min(1.0, np.mean(geometric_scores) if geometric_scores else 0.0)
    
    def _analyze_contrast_tonal(self, gray: np.ndarray) -> float:
        """Analyze contrast and tonal range."""
        # Calculate overall contrast
        min_val, max_val = np.min(gray), np.max(gray)
        contrast_range = (max_val - min_val) / 255.0
        
        # Calculate standard deviation (tonal variation)
        tonal_variation = np.std(gray) / 255.0
        
        # Calculate histogram distribution
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # Normalize
        
        # Check for good tonal distribution (not too concentrated)
        shadow_ratio = np.sum(hist[:85])  # 0-85 (shadows)
        midtone_ratio = np.sum(hist[85:170])  # 85-170 (midtones)
        highlight_ratio = np.sum(hist[170:])  # 170-255 (highlights)
        
        # Balanced distribution is good
        distribution_balance = 1.0 - np.std([shadow_ratio, midtone_ratio, highlight_ratio])
        
        # Combine contrast metrics
        contrast_score = (
            0.4 * contrast_range +
            0.3 * tonal_variation +
            0.3 * distribution_balance
        )
        
        return contrast_score
    
    def _analyze_focal_points(self, gray: np.ndarray) -> float:
        """Analyze focal points and subject emphasis."""
        # Use corner detection to find potential focal points
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.01, minDistance=30)
        
        if corners is None or len(corners) == 0:
            return 0.0
        
        height, width = gray.shape
        center_x, center_y = width // 2, height // 2
        
        # Analyze corner distribution and strength
        focal_scores = []
        for corner in corners:
            x, y = corner.ravel().astype(int)
            
            # Distance from center (closer to center is better for focal points)
            distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            center_score = 1.0 - (distance_from_center / max_distance)
            
            # Local contrast around the corner
            region_size = 20
            y1, y2 = max(0, y-region_size), min(height, y+region_size)
            x1, x2 = max(0, x-region_size), min(width, x+region_size)
            
            if y2 > y1 and x2 > x1:
                local_region = gray[y1:y2, x1:x2]
                local_contrast = np.std(local_region) / 255.0
            else:
                local_contrast = 0.0
            
            focal_score = 0.6 * center_score + 0.4 * local_contrast
            focal_scores.append(focal_score)
        
        # Return the best focal point score
        return max(focal_scores) if focal_scores else 0.0
    
    def _generate_grid_positions(self, img_w: int, img_h: int, crop_w: int, crop_h: int) -> List[Tuple[int, int]]:
        """Generate a grid of crop positions across the entire image."""
        positions = []
        
        # Create overlapping grid with step size to ensure good coverage
        step_x = crop_w // 2  # 50% overlap (less overlap for more diversity)
        step_y = crop_h // 2
        
        for y in range(0, img_h - crop_h + 1, step_y):
            for x in range(0, img_w - crop_w + 1, step_x):
                positions.append((x, y))
        
        return positions
    
    def _check_visual_element_presence(self, regions: List[Dict], crop_bbox: Tuple[int, int, int, int]) -> float:
        """Check if crop contains any interesting visual elements."""
        crop_x, crop_y, crop_w, crop_h = crop_bbox
        element_score = 0.0
        
        for region in regions:
            elem_x, elem_y, elem_w, elem_h = region['bbox']
            elem_center_x, elem_center_y = region['center']
            
            # Check if element center is within crop
            if (crop_x <= elem_center_x <= crop_x + crop_w and 
                crop_y <= elem_center_y <= crop_y + crop_h):
                # Bonus based on element importance
                element_score += region['score'] * 0.3
        
        return min(1.0, element_score)
    
    def _remove_overlapping_crops(self, crops: List[Dict], overlap_threshold: float = 0.125) -> List[Dict]:
        """Remove overlapping crops, keeping the ones with better composition."""
        if not crops:
            return crops
        
        diverse_crops = []
        for crop in crops:
            crop_x, crop_y, crop_w, crop_h = crop['bbox']
            is_overlapping = False
            
            for existing in diverse_crops:
                ex_x, ex_y, ex_w, ex_h = existing['bbox']
                
                # Calculate intersection
                x_left = max(crop_x, ex_x)
                y_top = max(crop_y, ex_y)
                x_right = min(crop_x + crop_w, ex_x + ex_w)
                y_bottom = min(crop_y + crop_h, ex_y + ex_h)
                
                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    union = crop_w * crop_h + ex_w * ex_h - intersection
                    overlap = intersection / union if union > 0 else 0
                    
                    if overlap > overlap_threshold:
                        is_overlapping = True
                        break
            
            if not is_overlapping:
                diverse_crops.append(crop)
        
        return diverse_crops
    


class MarkovChainBuilder:
    """Builds Markov chain transition matrix from crop relationships."""
    
    def __init__(self):
        self.transition_matrix = {}
        self.crop_features = {}
    
    def extract_features(self, image: np.ndarray, crop: Dict) -> Dict:
        """Extract features from a crop for Markov chain analysis."""
        x, y, w, h = crop['bbox']
        crop_image = image[y:y+h, x:x+w]
        
        # Color features
        mean_color = np.mean(crop_image, axis=(0, 1))
        color_std = np.std(crop_image, axis=(0, 1))
        
        # Brightness and contrast
        gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        
        # Texture features using local binary patterns
        try:
            lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
            texture_energy = np.var(lbp)
        except:
            texture_energy = 0
        
        return {
            'mean_color': mean_color.tolist(),
            'color_std': color_std.tolist(),
            'brightness': float(brightness.item() if hasattr(brightness, 'item') else brightness),
            'contrast': float(contrast.item() if hasattr(contrast, 'item') else contrast),
            'edge_density': float(edge_density.item() if hasattr(edge_density, 'item') else edge_density),
            'texture_energy': float(texture_energy.item() if hasattr(texture_energy, 'item') else texture_energy),
            'aspect_ratio': list(crop['aspect_ratio']),  # Convert tuple to list
            'region_type': crop['region_type'],
            'score': float(crop['score'].item() if hasattr(crop['score'], 'item') else crop['score'])
        }
    
    def calculate_similarity(self, features1: Dict, features2: Dict, spatial_distance: float = 0.0) -> float:
        """Calculate visual difference (inverted similarity) for contrast-driven transitions."""
        
        # Color difference (higher difference = higher probability)
        color_diff = np.linalg.norm(
            np.array(features1['mean_color']) - np.array(features2['mean_color'])
        )
        color_contrast = min(1.0, color_diff / 100)  # Normalize to 0-1
        
        # Brightness difference
        brightness_diff = abs(features1['brightness'] - features2['brightness'])
        brightness_contrast = min(1.0, brightness_diff / 100)
        
        # Contrast difference
        contrast_diff = abs(features1['contrast'] - features2['contrast'])
        contrast_contrast = min(1.0, contrast_diff / 50)
        
        # Edge density difference
        edge_diff = abs(features1['edge_density'] - features2['edge_density'])
        edge_contrast = min(1.0, edge_diff * 20)
        
        # Texture difference
        texture_diff = abs(features1['texture_energy'] - features2['texture_energy'])
        texture_contrast = min(1.0, texture_diff / 2000)
        
        # Spatial distance bonus (normalized to 0-1)
        distance_bonus = min(1.0, spatial_distance / 2000)  # Normalize distance
        
        # Weighted combination favoring visual differences and spatial distance
        visual_contrast = (
            0.30 * color_contrast +      # 30% - Color contrast
            0.20 * brightness_contrast + # 20% - Brightness contrast
            0.15 * contrast_contrast +   # 15% - Contrast difference
            0.10 * edge_contrast +       # 10% - Edge density contrast
            0.05 * texture_contrast +    # 5% - Texture contrast
            0.20 * distance_bonus        # 20% - Spatial distance bonus
        )
        
        return visual_contrast
    
    def build_transition_matrix(self, image: np.ndarray, crops: List[Dict]) -> Dict:
        """Build Markov chain transition matrix from crop features."""
        # Extract features for all crops
        self.crop_features = {}
        for i, crop in enumerate(crops):
            self.crop_features[i] = self.extract_features(image, crop)
        
        # Calculate transition probabilities
        n_crops = len(crops)
        self.transition_matrix = {}
        
        for i in range(n_crops):
            self.transition_matrix[i] = {}
            similarities = []
            
            for j in range(n_crops):
                if i != j:
                    # Calculate spatial distance between crop centers
                    center1 = crops[i]['region_center']
                    center2 = crops[j]['region_center']
                    spatial_distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                    
                    similarity = self.calculate_similarity(
                        self.crop_features[i], 
                        self.crop_features[j],
                        spatial_distance
                    )
                    similarities.append((j, similarity))
            
            # Normalize similarities to create probabilities
            total_similarity = sum(sim for _, sim in similarities)
            if total_similarity > 0:
                for j, sim in similarities:
                    self.transition_matrix[i][j] = sim / total_similarity
        
        return self.transition_matrix


class SequenceGenerator:
    """Generates photo sequences using the Markov chain."""
    
    def __init__(self, markov_builder: MarkovChainBuilder):
        self.markov_builder = markov_builder
    
    def generate_sequence(self, crops: List[Dict], sequence_length: int = 8, 
                         start_crop_idx: Optional[int] = None) -> List[int]:
        """Generate a sequence of crop indices using the Markov chain."""
        if not crops:
            return []
        
        n_crops = len(crops)
        
        # Choose starting crop
        if start_crop_idx is None:
            # Randomly select starting crop from top 50% of crops by score
            crops_sorted = sorted(range(n_crops), key=lambda i: crops[i]['score'], reverse=True)
            top_half = crops_sorted[:max(1, n_crops // 2)]  # At least 1 crop, top 50%
            start_crop_idx = random.choice(top_half)
        
        sequence = [start_crop_idx]
        used_crops = {start_crop_idx}  # Track used crops to avoid duplicates
        current_crop = start_crop_idx
        
        for _ in range(sequence_length - 1):
            # Get transition probabilities from current crop
            if current_crop in self.markov_builder.transition_matrix:
                transitions = self.markov_builder.transition_matrix[current_crop]
                
                # Filter out already used crops
                available_candidates = [crop for crop in transitions.keys() if crop not in used_crops]
                available_probabilities = [transitions[crop] for crop in available_candidates]
                
                if available_candidates and available_probabilities:
                    # Renormalize probabilities for available crops
                    total_prob = sum(available_probabilities)
                    if total_prob > 0:
                        normalized_probs = [p / total_prob for p in available_probabilities]
                        next_crop = np.random.choice(available_candidates, p=normalized_probs)
                    else:
                        # If all probabilities are zero, choose randomly
                        next_crop = random.choice(available_candidates)
                    
                    sequence.append(next_crop)
                    used_crops.add(next_crop)
                    current_crop = next_crop
                else:
                    # Fallback: choose random unused crop
                    unused_crops = [i for i in range(n_crops) if i not in used_crops]
                    if unused_crops:
                        next_crop = random.choice(unused_crops)
                        sequence.append(next_crop)
                        used_crops.add(next_crop)
                        current_crop = next_crop
                    else:
                        # If all crops used, break the sequence
                        break
            else:
                # Fallback: choose random unused crop
                unused_crops = [i for i in range(n_crops) if i not in used_crops]
                if unused_crops:
                    next_crop = random.choice(unused_crops)
                    sequence.append(next_crop)
                    used_crops.add(next_crop)
                    current_crop = next_crop
                else:
                    # If all crops used, break the sequence
                    break
        
        return sequence


class MarkovPhotoCropper:
    """Main class that orchestrates the entire photo cropping and sequencing process."""
    
    def __init__(self):
        self.analyzer = PhotoAnalyzer()
        self.crop_generator = CropGenerator()
        self.markov_builder = MarkovChainBuilder()
        self.sequence_generator = SequenceGenerator(self.markov_builder)
    
    def process_image(self, image_path: str, sequence_length: int = 4, 
                     output_dir: str = "output") -> Dict:
        """Process an image to generate a photo sequence."""
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to BGR if needed (for RGBA or grayscale)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        print(f"Processing image: {os.path.basename(image_path)}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Analyze image to find interesting regions
        regions = self.analyzer.find_interesting_regions(image)
        
        # Generate crops from regions
        crops = self.crop_generator.generate_crops(image, regions)
        print(f"Generated {len(crops)} diverse crops")
        
        # Build Markov chain
        transition_matrix = self.markov_builder.build_transition_matrix(image, crops)
        
        # Generate sequence
        sequence = self.sequence_generator.generate_sequence(crops, sequence_length)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate sequence images for collage (temporary, not saved individually)
        sequence_images = []
        for i, crop_idx in enumerate(sequence):
            x, y, w, h = crops[crop_idx]['bbox']
            crop_image = image[y:y+h, x:x+w]
            sequence_images.append(crop_image)
        
        # Create visualizations (collage and crop visualization)
        self._create_sequence_visualization(image, crops, sequence, output_dir)
        
        print(f"✅ Sequence complete! Results saved to: {output_dir}")
        
        return {
            'crops': crops,
            'sequence': sequence,
            'sequence_images': sequence_images
        }
    
    def _create_sequence_visualization(self, image: np.ndarray, crops: List[Dict], 
                                     sequence: List[int], output_dir: str):
        """Create a visualization showing the sequence of crops on the original image."""
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Draw all crop regions in light colors
        for i, crop in enumerate(crops):
            x, y, w, h = crop['bbox']
            color = (100, 100, 100)  # Gray for non-sequence crops
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis_image, str(i), (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Highlight sequence crops with bright colors
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
        
        for i, crop_idx in enumerate(sequence):
            crop = crops[crop_idx]
            x, y, w, h = crop['bbox']
            color = colors[i % len(colors)]
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 3)
            cv2.putText(vis_image, f"S{i}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Save visualization
        vis_path = os.path.join(output_dir, "sequence_visualization.png")
        cv2.imwrite(vis_path, vis_image)
        
        # Create a grid showing the sequence
        self._create_sequence_grid(image, crops, sequence, output_dir)
    
    def _create_sequence_grid(self, image: np.ndarray, crops: List[Dict], sequence: List[int], output_dir: str):
        """Create a professional horizontal collage of the sequence photos."""
        if not sequence:
            return
        
        # Generate sequence images directly from the original image
        sequence_images = []
        for crop_idx in sequence:
            x, y, w, h = crops[crop_idx]['bbox']
            crop_image = image[y:y+h, x:x+w]
            sequence_images.append(crop_image)
        
        # Resize all images to the same height for uniform presentation
        target_height = 400  # Standard height for collage
        resized_images = []
        for img in sequence_images:
            # Calculate new width maintaining aspect ratio
            aspect_ratio = img.shape[1] / img.shape[0]
            new_width = int(target_height * aspect_ratio)
            resized_img = cv2.resize(img, (new_width, target_height))
            resized_images.append(resized_img)
        
        # Calculate dimensions for professional collage
        border_width = 40  # Horizontal border around entire collage
        vertical_padding = 60  # Extra vertical padding above and below images
        image_spacing = 30  # Spacing between images
        total_width = sum(img.shape[1] for img in resized_images) + image_spacing * (len(resized_images) - 1) + border_width * 2
        total_height = target_height + vertical_padding * 2
        
        # Create white background with border
        collage = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
        
        # Place images horizontally with professional spacing
        x_offset = border_width
        for i, img in enumerate(resized_images):
            y_offset = vertical_padding  # Use vertical padding for top/bottom spacing
            
            # Place image on white background
            collage[y_offset:y_offset + img.shape[0], 
                   x_offset:x_offset + img.shape[1]] = img
            
            # Move to next position
            x_offset += img.shape[1] + image_spacing
        
        # Save professional collage
        stack_path = os.path.join(output_dir, "sequence_collage.png")
        cv2.imwrite(stack_path, collage)
        


def main():
    """Main function to run the MarkovPhotoCropper."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🎭 MarkovPhotoCropper - AI-Powered Photo Sequence Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python markov_photo_cropper.py photo.jpg
  python markov_photo_cropper.py photo.jpg -l 6 -o my_sequence
  python markov_photo_cropper.py photo.jpg --length 8 --output results
  python markov_photo_cropper.py --batch -o batch_results
        """
    )
    parser.add_argument("image_path", nargs='?', help="Path to the input image (or use --batch for all assets)")
    parser.add_argument("-l", "--length", type=int, default=4, help="Length of the generated sequence (default: 4)")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")
    parser.add_argument("--batch", action="store_true", help="Process all images in assets/ folder")
    
    args = parser.parse_args()
    
    # Create cropper instance
    cropper = MarkovPhotoCropper()
    
    print("MarkovPhotoCropper - AI-Powered Photo Sequence Generator")
    print("=" * 60)
    
    try:
        if args.batch:
            # Batch processing mode
            assets_dir = "assets"
            if not os.path.exists(assets_dir):
                print(f"❌ Assets directory '{assets_dir}' not found!")
                return 1
            
            # Find all image files in assets directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = []
            for file in os.listdir(assets_dir):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(assets_dir, file))
            
            if not image_files:
                print(f"❌ No image files found in '{assets_dir}' directory!")
                return 1
            
            print(f"📁 Found {len(image_files)} images to process...")
            
            # Process each image
            for i, image_path in enumerate(image_files, 1):
                print(f"\n🖼️  Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
                
                # Create subfolder for this image
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                subfolder = os.path.join(args.output, image_name)
                
                # Process the image
                result = cropper.process_image(image_path, args.length, subfolder)
                
                print(f"  ✅ Generated sequence of {len(result['sequence'])} photos")
                for j, crop_idx in enumerate(result['sequence']):
                    crop = result['crops'][crop_idx]
                    score_val = crop['score'].item() if hasattr(crop['score'], 'item') else crop['score']
                    print(f"    {j+1}. {crop['region_type'].title()} (score: {score_val:.3f})")
            
            print(f"\n🎉 Batch processing complete! All results saved to: {args.output}")
            
        else:
            # Single image processing mode
            if not args.image_path:
                print("❌ Please provide an image path or use --batch for batch processing")
                parser.print_help()
                return 1
            
            # Process the single image
            result = cropper.process_image(args.image_path, args.length, args.output)
            
            print(f"\n🎬 Generated sequence of {len(result['sequence'])} photos:")
            for i, crop_idx in enumerate(result['sequence']):
                crop = result['crops'][crop_idx]
                score_val = crop['score'].item() if hasattr(crop['score'], 'item') else crop['score']
                print(f"  {i+1}. {crop['region_type'].title()} (score: {score_val:.3f})")
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
