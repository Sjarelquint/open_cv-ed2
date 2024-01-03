import cv2

img1 = cv2.imread('D:/cv/form.jpg', 0)
h, w = img1.shape
img1 = cv2.resize(img1, (w // 2, h // 2))
img2 = cv2.imread('D:/cv/scanned-form.jpg', 0)
img2 = cv2.resize(img2, (w // 2, h // 2))


def orb1():
    orb = cv2.SIFT_create(1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # imKp1=cv2.drawKeypoints(img1,kp1,None)
    # imKp2=cv2.drawKeypoints(img2,kp2,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, 2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2
                              )
    # cv2.imshow('kp1', imKp1)
    # cv2.imshow('kp2', imKp2)
    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    cv2.imshow('img3', img3)
    cv2.waitKey(0)


def orb2():
    orb = cv2.ORB_create()
    keypoint1, descriptor1 = orb.detectAndCompute(img1, None)
    keypoint2, descriptor2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptor1, descriptor2)
    matches = sorted(matches, key=lambda x: x.distance)
    nMatches = 20
    img3 = cv2.drawMatches(img1, keypoint1, img2, keypoint2, matches[:nMatches], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('img3', img3)
    cv2.waitKey(0)


def orb3():
    orb = cv2.ORB_create(1000)
    keypoint1, descriptor1 = orb.detectAndCompute(img1, None)
    keypoint2, descriptor2 = orb.detectAndCompute(img2, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptor1, descriptor2, None)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * 0.1)
    matches = matches[:numGoodMatches]
    img3 = cv2.drawMatches(img1, keypoint1, img2, keypoint2, matches, None)
    cv2.imshow('img3', img3)
    cv2.waitKey(0)

def Flann():
    sift = cv2.SIFT_create()
    keypoints1, descriptor1 = sift.detectAndCompute(img1,None)
    keypoints2, descriptor2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    nKDtrees = 5
    nLeafChecks = 50
    nNeighbors = 2
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=nKDtrees)
    searchParams = dict(checks=nLeafChecks)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(descriptor1, descriptor2, k=nNeighbors)
    matchesMask = [[0, 0] for i in range(len(matches))]
    testRatio = 0.75  # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < testRatio * n.distance:
            matchesMask[i] = [1, 0]
    drawParams = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                      matchesMask=matchesMask, flags=cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, None, **drawParams)
    cv2.imshow('img3', img3)
    cv2.waitKey(0)


if __name__ == '__main__':
    orb1()
    # orb2()
    # orb3()
    # Flann()
