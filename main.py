import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import imageio_ffmpeg
import subprocess as sp

THRESHOLD = 0.75 # threshold for matching template to frame

def resize_image(img, dims = (1920, 1080)):
    return cv.resize(img, dims, interpolation=cv.INTER_AREA)

def remove_non_red(img):
    """
    Removes all non-red colour from image
    Credit: stackoverflow.com/users/1390022/derricw
    """
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv.inRange(img_hsv, lower_red, upper_red)

    mask = mask0 + mask1

    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0

    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0

    return output_hsv

def process_frame_img(img, template, method = cv.TM_SQDIFF_NORMED, threshold = THRESHOLD):
    """
    This function processes a frame and looks for a match between the frame and the template.
    Returns image
    Currently for testing purposes
    Credit: https://github.com/learncodebygaming
    """
    # Resize image
    img = resize_image(img)

    # Removing non-red colours - much better accuracy when matching
    img = remove_non_red(img)
    template = remove_non_red(template)

    result = cv.matchTemplate(img, template, method)

    # this logic could be wrong, haven't tested with other cv.methods
    if method == cv.TM_SQDIFF_NORMED:
        locations = np.where(result <= threshold)
    else:
        locations = np.where(result >= threshold)

    locations = list(zip(*locations[::-1]))

    if locations:
        print('Found template.')

        template_w = template.shape[1]
        template_h = template.shape[0]
        line_color = (0, 255, 0)
        line_type = cv.LINE_4

        # Loop over all the locations and draw their rectangle
        for loc in locations:
            # Determine the box positions
            top_left = loc
            bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
            # Draw the box
            cv.rectangle(img, top_left, bottom_right, line_color, line_type)

        # cv.imwrite(f'test.jpg', img)
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])
        plt.show()

    else:
        print('Template not found.')

def process_frame(img, template, method = cv.TM_SQDIFF_NORMED, threshold = THRESHOLD):
    """
    This function processes a frame and looks for a match between the frame and the template.
    Returns bool
    Credit: https://github.com/learncodebygaming
    """
    # Resize image
    img = resize_image(img)
    
    # Removing non-red colours - much better accuracy when matching
    img = remove_non_red(img)
    template = remove_non_red(template)

    result = cv.matchTemplate(img, template, method)

    # this logic could be wrong, haven't tested with other cv.methods
    if method == cv.TM_SQDIFF_NORMED:
        locations = np.where(result <= threshold)
    else:
        locations = np.where(result >= threshold)

    locations = list(zip(*locations[::-1]))

    if locations:
        return True
    else:
        return False
    
def get_frame(vidPath, time):
    """
    Get specific frame of video. For testing purposes
    """
    video = cv.VideoCapture(vidPath)

    vidFps = video.get(cv.CAP_PROP_FPS)
    frameNum = time*vidFps

    video.set(cv.CAP_PROP_POS_FRAMES, frameNum)

    success, frame = video.read()

    if success:
        video.release()
        return frame
    else:
        video.release()
        print('Frame not found.')

def locate_clip_boundaries(vid, skipFrames, template):
    """
    Selects every n frames from video and template matches using the process_frame function.
    Returns a list, boundaries, of the frame numbers where process_frame returned True -- frames where there is a player kill in the feed.
    :param cv.VideoCapture vid: OpenCV VideoCapture video object
    :param int skipFrames: Number of frames to skip
    """
    frameNum = 0
    count = 0 # using this to skip frames

    global numFrames, fps, duration # declaring global variables to use in other function
    numFrames = vid.get(cv.CAP_PROP_FRAME_COUNT)
    fps = vid.get(cv.CAP_PROP_FPS)
    duration = numFrames / fps

    boundaries = [] # frame positions of the kills

    while (True):
        success, frame = vid.read()

        if success:
            result = process_frame(frame, template)

            if count > numFrames:
                break

            if result:
                boundaries.append(vid.get(cv.CAP_PROP_POS_FRAMES)) # adds current frame number to list

            count += skipFrames # this will skip every 60 frames
            vid.set(cv.CAP_PROP_POS_FRAMES, count)
        else:
            break

    frameNum = frameNum + 1
    vid.release()

    return boundaries

def clip_video(vidPath, boundaries, padding, vidSuffix):
    """
    Clipping the video between the first and last boundaries, + extra padding (seconds)
    """
    if not boundaries:
        print('No timestamps found')
    else:
        # Using min() and max() to ensure the times aren't outside the bounds of the video
        startTime = max(boundaries[0]/fps - padding, 0)
        endTime = min(boundaries[-1]/fps + padding, duration)

        ffmpeg_extract_subclip(vidPath, startTime, endTime, targetname=f'output/{vidPath.split("/")[-1].strip(".mp4")}{vidSuffix}.mp4')

def process_video(filePath, padding, skipFrames, vidSuffix, template):
    """
    Template match every n frames of a video to locate the frame positions of the kills.
    Crop the video using the boundaries found.
    Returns a clipped mp4.
    :param str filePath: Path to the video to be clipped
    :param int padding: Number of seconds padding around the clip
    :param int skipFrames
    """
    video = cv.VideoCapture(filePath)
    
    boundaries = locate_clip_boundaries(video, skipFrames, template)

    clip_video(filePath, boundaries, padding, vidSuffix)

def main():
    template = cv.imread('assets/killfeed_template_3.jpg')
    padding = 2
    skipFrames = 60
    vidSuffix = '_clipped'

    inputDir = 'input'
    dirFiles = os.listdir(inputDir)
    vidFiles = [fName for fName in dirFiles if fName.endswith('.mp4')]

    for i, vid in enumerate(vidFiles):
        process_video(f'{inputDir}/{vid}', padding, skipFrames, vidSuffix, template)
        
        initSize = os.path.getsize(f"input/{vid}")//1000000

        try:
            finalSize = os.path.getsize('output/' + vid.strip('.mp4') + '_clipped.mp4')//1000000
            print(f'\n{i+1}/{len(vidFiles)} completed:\n{vid} \n{initSize}MB -> {finalSize}MB')
        except FileNotFoundError:
            print('No output clip created.')

if __name__ == "__main__":
    main()