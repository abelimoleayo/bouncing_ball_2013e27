# ENGR 27 Final Project
#
# Imoleayo Abel, Eliza Bailey, and Danielle Sullivan.


import numpy as np
import cv, cv2, Tkinter, sys, cvk2, math, struct, time
from random import choice

global win_width, win_height

###############################################################################
# Camera Device Verification

# Try to get an integer argument:
try:
    device = int(sys.argv[1])
    del sys.argv[1]
except (IndexError, ValueError):
    device = 0

# If we have no further arguments, open the device. Otherwise, get the filename.
if len(sys.argv) == 1:
    capture = cv2.VideoCapture(device)
    if capture:
        print 'Opened device number', device
else:
    capture = cv2.VideoCapture(sys.argv[1])
    if capture:
        print 'Opened file', sys.argv[1]
# Bail if error.
if not capture:
    print 'Error opening video capture!'
    sys.exit(1)

#################################################################################
# Get screen dimensions

screen = Tkinter.Tk()
win_width = screen.winfo_screenwidth()
win_height = screen.winfo_screenheight()
screen.destroy() 

cv2.namedWindow('Final Project', cv2.cv.CV_WINDOW_NORMAL)
cv2.cv.MoveWindow('Final Project',0,0)
cv2.cv.ResizeWindow('Final Project',win_width,win_height)

##################################################################################
# Prepare video output

fps = 6
fourcc, ext = (struct.unpack('i', 'DIVX')[0], 'avi')
videofilename = 'adeFinalProject.' + ext
videofilename2 = 'adeFinalProjectWarped.' + ext
writer = cv2.VideoWriter(videofilename, fourcc, fps, (win_width, win_height))
writerWarped = cv2.VideoWriter(videofilename2, fourcc, fps, (win_width, win_height))
if (not writer) or (not writerWarped):
    print 'Error opening writer'
else:
    print 'Opened', videofilename, 'and', videofilename2, 'for output.'

######################################################################################
# Ball class definition

class Ball(object):
    def __init__(self, radius, color):
        self.radius  = radius
        x = int(((win_width - 2*self.radius) * np.random.random_sample()) + self.radius)
        y = int(((0.5*win_height - self.radius) * np.random.random_sample()) + self.radius)
        self.pos = np.array([x,y],dtype='f')
        self.vel = (np.random.rand(2)*2-1)*750
        self.color = color
        self.k_restitution = 0.4#0.8
        self.k_friction = 0.99
        self.k_collision = 0.05
        self.k_rvel = 1
        self.dists_prev = None
        self.doffs_prev = None

#######################################################################################
# Some helper functions

def sample(img, pos):

    sample_subpix = None   
    sample_subpix = cv2.getRectSubPix(img, (3,3), tuple(pos), 
                                           sample_subpix)
    dist_to_nearest = sample_subpix[1,1]
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='f')/8
    ky = kx.transpose()
    gx = (sample_subpix*kx).sum()
    gy = (sample_subpix*ky).sum()
    return dist_to_nearest, gx, gy

def normalize(v):
    vn = np.linalg.norm(v)
    if vn:
        v /= vn
    return v

def extractSubimage(img, ball, dst, default=255):

    wrad = int(1.5*ball.radius+20)
    wsz = 2*wrad + 1

    [ix,iy] = np.round(ball.pos).astype(int)

    ix0 = ix-wrad
    ix1 = ix0+wsz
    iy0 = iy-wrad
    iy1 = iy0+wsz

    ax0 = max(0, min(ix0, win_width))
    ax1 = max(0, min(ix1, win_width))
    ay0 = max(0, min(iy0, win_height))
    ay1 = max(0, min(iy1, win_height))

    sy0 = ay0-iy0
    sy1 = sy0 + ay1-ay0
    sx0 = ax0-ix0
    sx1 = sx0 + ax1-ax0

    dst[:] = default
    dst[sy0:sy1, sx0:sx1] = img[ay0:ay1, ax0:ax1]
    return dst

#####################################################################
# Variable Definitions

white = (255,255,255)
black = (0,0,0)
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))

################################################################################

# check if there's a saved homography matrix, oherwise, find homography
try:
    H = np.load("H.npy")
except:
    # generate points for homography
    tol = 30
    p1 = (tol,tol)
    p2 = (win_width/2, tol)
    p3 = (win_width-tol, tol)
    p4 = (tol, win_height/2)
    p5 = (win_width/2, win_height/2)
    p6 = (win_width - tol, win_height/2)
    p7 = (tol, win_height - tol) 
    p8 = (win_width/2, win_height - tol)
    p9 = (win_width - tol, win_height - tol)
    p10 = (win_width/4 + tol/2, win_height/4 + tol/2)
    p11 = (3*win_width/4 - tol/2, win_height/4 + tol/2)
    p12 = (win_width/4 + tol/2, 3*win_height/4 - tol/2)
    p13 = (3*win_width/4 - tol/2, 3*win_height/4 - tol/2)
    projectedPoints = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]

    # initialize variables defined within if-blocks so that they have a global scope
    reference_background = 0
    frame_grey = 0
    mask = 0
    instance_pos = []
    n_instance = 8          # number of frames in which the position of a blob does not change before
                            # we consider it an interesting object
    cameraPoints = []       # array of position of projected points
    n_frame = 1

    backGround = np.empty((win_height,win_width),dtype='uint8') # image to be projected
    while 1:
        cv2.rectangle(backGround,(0,0),(win_width,win_height),white,-1) # white background
        if n_frame <= 21:
            # do nothing with captured frame to make sure camera settles 
            cv2.imshow('Final Project',backGround)
            cv2.waitKey(250)
            ok, frame = capture.read()
            if n_frame == 21:
                # set reference background
                ref_grey = np.empty((frame.shape[0],frame.shape[1]), 'uint8')                       
                cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY, ref_grey)
                reference_background = ref_grey
        else:
            raw_index = n_frame - 22
            index = raw_index/n_instance
            instance = raw_index%n_instance
            # break after 13th projected point has been registered
            if index > 12:
                cv2.imshow('Final Project',backGround)
                break

            # make filled black circle
            cv2.circle(backGround, (projectedPoints[index]), 20, black, -1, cv2.CV_AA)
            # update system status
            cv2.imshow('Final Project',backGround)      # project circle
            cv2.waitKey(1000/(n_instance))              # wait a few seconds
            ok, frame = capture.read()                  # capture view with camera

            # convert frame to grayscale
            frame_grey = np.empty((frame.shape[0],frame.shape[1]), 'uint8')   
            cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY, frame_grey)
            # get eroded thresholded difference image
            diff = cv2.absdiff(frame_grey,reference_background)
            mask = np.zeros(diff.shape, 'uint8') 
            cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY, mask)
            mask_erode = cv2.erode(mask,kernel=element,iterations=1)
            temp = mask_erode.copy()            # temp copy for getting contour info
            
            contours = cv2.findContours(temp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            areas = []          # area of blobs in current instance
            pos = []            # centroid of bloba in current instance
            for i in range(len(contours[0])):
                try:
                    info = cvk2.getcontourinfo(contours[0][i])
                    areas.append(info['area'])
                    pos.append(cvk2.a2ti(info['mean']))
                except:
                    pass
            # if more than one blob is found, save the centroid of the maximum area blob
            if len(areas) > 0:
                instance_pos.append(pos[areas.index(max(areas))])
            # after enough instances of the same view, check if the centroid of the maximum
            # area blob changes beyond a given threshold
            if instance == (n_instance-1):
                dot_errorThreshold = 20
                dists = []   # array of distances between centroids in adjacent instances
                for j in range(len(instance_pos)):
                    dists.append(math.sqrt(sum([(p-q)**2 for p,q in zip(instance_pos[j],
                                                          instance_pos[(j+1)%len(instance_pos)])])))
                # if there is no consistency in centroid of max area blobs in n_instance frames
                if (len(dists) == 0) or (max(dists) > dot_errorThreshold):
                    instance_pos = []                   # empty array of centroid of max area blobs
                    n_frame -= instance                 # rewind frame count index
                    k = cv2.waitKey(5)
                    # Check for ESC hit:
                    if k % 0x100 == 27:
                        break
                    continue    # ignore rest of loop and restart assertion of current projected 
                                # circle's centroid
                # if there's consistency in n_instance frames, compute average centroid location
                avg_pos = tuple([sum(p)/len(p) for p in zip(*instance_pos)])
                cameraPoints.append(avg_pos)        # append to camera points
                instance_pos = []

        k = cv2.waitKey(5)
        # Check for ESC hit:
        if k % 0x100 == 27:
            break
        n_frame += 1

    # get homography matrix 
    H = cv2.findHomography(np.array(cameraPoints,dtype=np.float32), \
                           np.array(projectedPoints,dtype=np.float32),0)[0]

    np.save('H',H)
    reference = cv2.warpPerspective(reference_background, H, (win_width,win_height))
    reference = cv2.imwrite("reference.png",reference) # warped empty screen

################################################################################
# More variable definitions

gravity = np.array([0,400])
dt_msec = 16
delay_msec = 4
updates_per_frame = 4
frame_dt = dt_msec * 1e-2
update_dt = frame_dt/updates_per_frame

dists = None
dists_smoothed = None
sample_subpix = None
dists_prev = None

backGround = np.empty((win_height,win_width,3),dtype='uint8') # image to be projected
reference = cv2.imread("reference.png",cv2.CV_LOAD_IMAGE_GRAYSCALE)
combined = np.empty_like(reference)                     # to be used later

Balls = []
n_balls = 4
for i in range(n_balls):
    ball = Ball(90,black)
    Balls.append(ball)

##################################################################################

tol = 10  # width of bounding box
while 1:

    # bounding box
    cv2.rectangle(backGround,(0,0),(win_width,win_height),white,-1)
    cv2.rectangle(backGround,(tol,tol),(win_width-tol,win_height-tol),black,-1)
    cv2.rectangle(backGround,(2*tol,2*tol),(win_width-2*tol,win_height-2*tol),white,-1)

    ok, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_warped = cv2.warpPerspective(gray, H, (win_width,win_height))
    diff = cv2.absdiff(gray_warped,reference)

    pre_mask = np.zeros(diff.shape, 'uint8')
    mask = np.zeros(diff.shape, 'uint8')

    # 2-Stage Thresholding Process
    cv2.threshold(diff, 250, 255, cv2.THRESH_TOZERO_INV, pre_mask) # remove balls
    cv2.threshold(pre_mask, 120, 255, cv2.THRESH_BINARY_INV, mask) # isolate obstruction
    human = cv2.dilate(mask,kernel=element,iterations=6) # morphological operation

    bgrndProc = backGround.copy()    # copy of background with only border
    # draw 
    for ball in Balls:
        cv2.circle(bgrndProc, tuple(np.round(ball.pos).astype(int)), 
                                        ball.radius, black, -1, cv2.CV_AA)
   
    # update with small timesteps multiple times
    for i in range(updates_per_frame):


        for ball in Balls:
            bgrndProc2 = bgrndProc.copy()
            epsilon = 1
            # cover up circle in consideration
            cv2.circle(bgrndProc2, tuple(np.round(ball.pos).astype(int)),
                                            ball.radius+epsilon, white, -1, cv2.CV_AA)
            bgrndProcGray = cv2.cvtColor(bgrndProc2, cv2.COLOR_RGB2GRAY)
            # combine obstruction and other circles
            np.bitwise_and(bgrndProcGray, human, combined)

            wrad = int(1.5*ball.radius+20)
            wsz = 2*wrad+1
            subimage = 255*np.ones((wsz,wsz),dtype='uint8')
            offs = np.array([wrad, wrad],dtype='f')
            ipos = tuple(np.round(ball.pos).astype(int))
            offs = np.array([wrad, wrad],dtype='f')
            doffs = offs-ipos

            # get rectangular section around circle
            extractSubimage(combined, ball, subimage)
            dists = cv2.distanceTransform(subimage, cv.CV_DIST_L2, cv.CV_DIST_MASK_PRECISE)
            dists_smoothed = cv2.GaussianBlur(dists[0], (0,0), 0.5)

            if (ball.dists_prev is None) and (i==0):
                ball.dists_prev = dists[0].copy()
                ball.doffs_prev = doffs.copy()

            vnew = ball.vel + update_dt*gravity
            ball.pos += 0.5*update_dt*(ball.vel + vnew)
            ball.vel = vnew

            dist_to_nearest, gx, gy = sample(dists_smoothed, ball.pos+doffs)

            if dist_to_nearest < ball.radius+epsilon:

                normal = normalize(np.array(( gx, gy )))

                dprev, gpx, gpy = sample(ball.dists_prev, ball.pos+ball.doffs_prev)
                nprev = normalize(np.array( (gpx, gpy) ))
                fvel = -ball.k_rvel * (dist_to_nearest*normal - dprev * nprev) / frame_dt

                rvel = ball.vel - fvel

                proj = np.dot(normal, rvel)
                ball.pos += ball.k_collision*dist_to_nearest*normal
                if proj < 0:
                    rvel_normal = normal * proj
                    rvel_tangent = rvel - rvel_normal
                    rvel = ball.k_friction*rvel_tangent - ball.k_restitution*rvel_normal
                    ball.vel = rvel + fvel
            r = ball.radius
            ball.pos = np.maximum((2*tol+r,2*tol+r), np.minimum(ball.pos, (win_width-2*tol-r, win_height-2*tol-r)))

            if i==(updates_per_frame-1):
                pos_int = tuple(np.round(ball.pos).astype(int))
                cv2.circle(backGround, pos_int, ball.radius, ball.color, -1, cv2.CV_AA)
                ball.dists_prev[:] = dists_smoothed
                ball.doffs_prev[:] = doffs

    cv2.imshow('Final Project', backGround)

    # write to output video files
    if writer and writerWarped:
        writer.write(frame)
        warped_frame = cv2.warpPerspective(frame, H, (win_width,win_height))
        writerWarped.write(warped_frame)

    k = cv2.waitKey(1)#50*(dt_msec - delay_msec))
    if k == 27:
        break
