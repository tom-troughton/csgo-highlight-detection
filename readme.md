# CSGO Highlight Clipper

An automated tool that uses computer vision to detect and extract kill feed moments from Counter-Strike: Global Offensive (CSGO) gameplay recordings.

## Overview

This tool analyzes CSGO gameplay videos to automatically detect when kills occur by identifying the kill feed notifications in the top-right corner of the screen. It then clips the video to create highlight reels containing only the kill sequences, significantly reducing video size while preserving the important moments.

## How It Works

1. The program scans through video files frame-by-frame (skipping frames for efficiency)
2. Uses OpenCV template matching to detect the red kill feed notifications
3. Records the timestamps where kills are detected
4. Clips the original video to include only the kill sequences with configurable padding

## Features

- Processes multiple video files in batch
- Configurable frame skip rate for performance optimization
- Adjustable padding around highlight moments
- Processes frames to improve detection accuracy
- Outputs compressed highlight clips

## Requirements

- OpenCV (cv2)
- NumPy
- Matplotlib
- MoviePy
- imageio-ffmpeg

## Usage

1. Place your CSGO gameplay recordings in the `input/` directory
2. Ensure you have a kill feed template image in `assets/killfeed_template_3.jpg`
3. Run the script: `python main.py`
4. Processed clips will be saved to the `output/` directory

## Configuration

You can adjust the following parameters in the code:
- `THRESHOLD`: Confidence threshold for template matching (default: 0.75)
- `padding`: Seconds of footage to keep before/after kills (default: 2)
- `skipFrames`: Number of frames to skip between checks (default: 60)
- Video dimensions (default: 1920x1080)
