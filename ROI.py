import cv2
import numpy as np

# -------------------- Parameters --------------------
blurSigma = 1.0
otsuThreshold = 0

freqThreshold = 10

neighborDistance = 50

# --------- Non-valley suppression function ----------
angle = np.array([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5])
angle = angle * np.pi / 180

sin = np.sin(angle)
cos = np.cos(angle)

d = np.array([neighborDistance, 0])
delta = np.zeros((16, 2))

for i in range(16):
    delta[i] = np.dot(np.array([[cos[i], -sin[i]], [sin[i], cos[i]]]), d)

def suppression(img, candidate):
    rows, cols = img.shape
    newCandidate = np.zeros((0, 2))
    
    for c in candidate:
        # Get neighbor pixels' value
        neighbor = np.round(c + delta).astype(np.int64)

        position = (neighbor[:, 1] >= 0) & (neighbor[:, 1] < rows) & (neighbor[:, 0] >= 0) & (neighbor[:, 0] < cols)
        neighbor = neighbor[position]
        value = img[neighbor[:, 1], neighbor[:, 0]]
        
        # Threshold the number of non-hand region pixels
        count = np.sum(value == 0)
        if count <= 7:
            newCandidate = np.append(newCandidate, c.reshape(1, 2).astype(np.int64), axis = 0)

    return newCandidate

# -------------- ROI extraction function -------------
def extractROI(original):
    # Reduce noise (Gaussian blur)
    blurred = cv2.GaussianBlur(original, (0, 0), blurSigma)

    # Lighten finger knuckles (logarit transform)
    c = 255 / np.log(np.max(blurred) + 1)
    enhanced = (c * np.log(blurred + 1)).astype(np.uint8)

    # Find hand region (Otsu threshold)
    retval, thresholded = cv2.threshold(enhanced, otsuThreshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find hand boundary (the largest area contour)
    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = [cv2.contourArea(contour) for contour in contours]
    
    order = np.argmax(area)
    boundary = contours[order]

    # Find center of the boundary
    moments = cv2.moments(boundary)
    center = [moments['m10'] // moments['m00'], moments['m01'] // moments['m00']]

    # Calculate distances from the boundary to its center
    boundary = boundary[:, 0, :]
    distances = np.sqrt(np.sum((boundary - center) ** 2, axis = 1)).reshape(-1)

    # Remove high frequency in distances (low-pass filter)
    frequency = np.fft.rfft(distances)
    filtered = np.concatenate([frequency[:freqThreshold], 0 * frequency[freqThreshold:]])
    smoothed = np.fft.irfft(filtered)

    # Find hand valley candidates (local minima)
    derivation = np.diff(smoothed)
    zeroCrossing = np.diff(np.sign(derivation)) / 2
    candidates = boundary[np.where(zeroCrossing > 0)[0]]

    # Non-valley suppression (CHVD algorithm)
    candidates = suppression(thresholded, candidates)

    # Get 1-st and 3-rd valley
    order = np.argsort(candidates[:, 0])
    candidates = candidates[order]
    candidates = candidates[:3]

    order = np.argsort(candidates[:, 1])
    valley = candidates[[order[0], order[2]]]

    # Rotate image
    valley0, valley1 = valley
    phi = - 90 + np.arctan2((valley1 - valley0)[1], (valley1 - valley0)[0]) * 180 / np.pi

    R = cv2.getRotationMatrix2D(tuple(center), phi, 1)
    rotated = cv2.warpAffine(original, R, original.shape[::-1])

    valley0 = (np.dot(R[:, :2], valley0) + R[:, -1]).astype(np.int)
    valley1 = (np.dot(R[:, :2], valley1) + R[:, -1]).astype(np.int)

    # Get ROI
    rect0 = (valley0[0] + 2 * (valley1[1] - valley0[1]) // 6, valley0[1] - (valley1[1] - valley0[1]) // 6)
    rect1 = (valley1[0] + 10 * (valley1[1] - valley0[1]) // 6, valley1[1] + (valley1[1] - valley0[1]) // 6)

    roi = rotated[rect0[1]:rect1[1], rect0[0]:rect1[0]]

    return roi