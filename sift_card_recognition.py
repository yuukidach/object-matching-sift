# -*- coding utf-8 -*-

import cv2
import numpy as np

def get_src():
    # 获取视频并截获第一张图片
    cap  = cv2.VideoCapture('basic_test.mp4')
    _, src_raw = cap.read()
    src_gray = cv2.cvtColor(src_raw, cv2.COLOR_BGR2GRAY)  
    cap.release()  

    # 先均衡化图像，增强对比度，然后检测边缘
    src = cv2.equalizeHist(src_gray)
    src = cv2.Canny(src, 10, 240)

    # 检测图中的白点，便于后续进行最小矩形检测
    height, width = src.shape
    points = []

    for i in range(width):
        for j in range(height):
            if (src[j, i] != 0):
                points.append([i, j])

    points = np.array(points)

    '''
    # 用最小矩形包裹canny算子检测出的边缘
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 查看找到的区域是否符合要求
    # cv2.drawContours(src_raw, [box], 0, (0, 0, 255), 2)
    '''

    x, y, w, h = cv2.boundingRect(points)
    src_gray = src_gray[y-10:y+h+10, x+10:x+w+35]      # 对截取区域增加偏差值，增加匹配正确率

    return src_gray


# 从视频中获取单张图片
def get_target(frame_pos):
    cap = cv2.VideoCapture('basic_test.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos-1)
    res, target = cap.read()
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) 
    cap.release() 

    return target


if __name__ == "__main__":
    src = get_src()
    
    cap = cv2.VideoCapture('basic_test.mp4')
    res, target = cap.read()
    while (res == True):
        res, target_raw = cap.read()
        target = cv2.cvtColor(target_raw, cv2.COLOR_BGR2GRAY) 
    
        '''
        cv2.imshow("test", src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        sift = cv2.xfeatures2d.SIFT_create()
        kpts1, descs1 = sift.detectAndCompute(src, None)
        kpts2, descs2 = sift.detectAndCompute(target, None)

        FLANN_INDEX_KDTREE = 1
        matcher = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_KDTREE, trees = 4), {})

        # 选取匹配最相近的两个点
        matches = matcher.knnMatch(descs1, descs2, 2)
        # 按距离排序
        matches = sorted(matches, key = lambda x:x[0].distance)
        # 判断最优和次优匹配比率是否足够低，够低则匹配成功
        good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]

        res_img = target_raw.copy()

        # 当有足够的健壮匹配点对（至少4个）时
        MIN_MATCH_COUNT = 4
        if len(good) >= MIN_MATCH_COUNT:
            # 从匹配中提取出对应点对
            # (queryIndex for the small object, trainIndex for the scene )
            src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            # find homography matrix in cv2.RANSAC using good match points
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # 计算图1的畸变，也就是在图2中的对应的位置。
            h, w = src.shape[:2]
            pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, M)
            # 绘制边框
            cv2.polylines(res_img, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

        '''
        # 画出匹配的对应点
        matched = cv2.drawMatches(src,kpts1,canvas,kpts2,good,None)#,**draw_params)

        # 将找到的结果剪裁下来，并变换到原图视角
        h,w = src.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
        found = cv2.warpPerspective(target,perspectiveM,(w,h))
        '''

        cv2.imshow("result", res_img)
        if cv2.waitKey(1) &0xFF ==ord('q'):  #按q键退出
    	    break

    cap.release() 
    cv2.destroyAllWindows()





