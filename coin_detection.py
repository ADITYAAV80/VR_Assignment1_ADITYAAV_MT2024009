import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import sharpen,read_image,downsample_images
class HoughCoinDetector:
    def __init__(self, downsample_size=512, gaussian_kernel=(3, 3), morph_kernel_size=(5,5), dist_thresh_factor=0.8):
        self.downsample_size = downsample_size
        self.gaussian_kernel = gaussian_kernel
        self.morph_kernel_size = morph_kernel_size
        self.dist_thresh_factor = dist_thresh_factor
    
    def cannyplots(self, image):
        fig, ax = plt.subplots(2, 6, figsize=(18, 4))
        for i in range(11):
            edges_img = cv2.Canny(image, i * 30, i * 30 + 60)
            row, col = divmod(i, 6)
            ax[row, col].imshow(edges_img, cmap='gray')
            ax[row, col].set_title(f'Canny {i * 30}-{i * 30 + 60}')
            ax[row, col].axis('off')
        ax[1, 5].imshow(image, cmap='gray')
        ax[1, 5].set_title('Original Image')
        ax[1, 5].axis('off')
        plt.tight_layout()
        plt.show()
    
    def detect_with_Hough(self, image, canny_higher, minDist=None, threshold=30, minRadius=None, maxRadius=None):
        
        if not minDist: minDist = image.shape[0] // 8
        if not minRadius: minRadius = image.shape[0] // 15
        if not maxRadius: maxRadius = image.shape[0] // 8
        if not threshold: threshold = 30

        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, minDist, param1=canny_higher, param2=threshold, minRadius=minRadius, maxRadius=maxRadius)
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            for x, y, r in circles:
                cv2.circle(image, (x, y), r, (0, 0, 0), 2)
                cv2.circle(image, (x, y), 0, (0, 0, 0), 3)
        return image, len(circles) if circles is not None else 0
    
    def process_hough(self, path):
        
        img = read_image(path)
        cv2.imwrite('output/hough_original.jpg', img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('output/hough_gray.jpg', gray)

        gray = downsample_images(gray)
        cv2.imwrite('output/hough_downsampled_gray.jpg', gray)

        lapacian, sharpened_image = sharpen(gray)
        cv2.imwrite('output/hough_laplacian.jpg', lapacian)
        cv2.imwrite('output/hough_sharpened.jpg', sharpened_image)

        image_blurred = cv2.GaussianBlur(sharpened_image, self.gaussian_kernel, 0)
        cv2.imwrite('output/hough_blurred.jpg', image_blurred)

        self.cannyplots(image_blurred)
        canny_lower, canny_higher = map(int, input("Enter lower and higher threshold: ").split())

        minDist = int(input("Enter minDist 0 for auto : "))
        threshold = int(input("Enter threshold 0 for auto : "))
        minRadius = int(input("Enter minRadius 0 for auto : "))
        maxRadius = int(input("Enter maxRadius 0 for auto : "))

        hough_image, coins = self.detect_with_Hough(image_blurred, canny_higher, minDist, threshold, minRadius, maxRadius)
        cv2.imwrite('output/hough_result.jpg', hough_image)
        print("Total coins detected:", coins)
        


class WatershedProcessor:
    def __init__(self, downsample_size=512, gaussian_kernel=(3, 3), morph_kernel_size=(5, 5), dist_thresh_factor=0.8, morphology_iterations=1,erode_iterations=3):
        self.downsample_size = downsample_size
        self.gaussian_kernel = gaussian_kernel
        self.morph_kernel_size = morph_kernel_size
        self.dist_thresh_factor = dist_thresh_factor
        self.morphology_iterations = morphology_iterations
        self.erode_iterations = erode_iterations
    
    def read_image(self, path):
        return cv2.imread(path)
    
    def downsample_images(self, image):
        while image.shape[0] > self.downsample_size or image.shape[1] > self.downsample_size:
            image = cv2.pyrDown(image)
        return image
    
    def apply_high_pass_filter(self, image):
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpened = cv2.addWeighted(image, 1.5, cv2.convertScaleAbs(laplacian), -0.5, 0)
        return laplacian, sharpened
    
    def apply_threshold(self, image):
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        print("Optimal_Threshold:", ret)
        return thresh
    
    def apply_morphology(self, image):
        kernel = np.ones(self.morph_kernel_size, np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=self.morphology_iterations)
    
    def compute_distance_transform(self, image):
        dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 3)
        ret, sure_fg = cv2.threshold(dist_transform, self.dist_thresh_factor * dist_transform.max(), 255, cv2.THRESH_BINARY)
        return dist_transform,np.uint8(sure_fg)
    
    def erode(self,filled_image):
        sure_bg = cv2.erode(filled_image,self.morph_kernel_size,self.erode_iterations)
        return sure_bg
    
    def apply_watershed(self, img, sure_fg, filled_image):
        difference_image = cv2.subtract(filled_image, sure_fg)
        total_coins, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[difference_image == 255] = 0

        cv2.watershed(img, markers)

        # Create output image
        segmented_img = img.copy()
        segmented_img[markers == -1] = [0, 0, 255]  # Mark boundaries in red

        return (total_coins - 1), segmented_img

    
    def process_watershed(self, path):
        
        img = read_image(path)
        cv2.imwrite('output/watershed_original.jpg', img)

        img = downsample_images(img)
        cv2.imwrite('output/watershed_downsampled.jpg', img)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('output/watershed_gray.jpg', gray)
        
        laplacian, sharpened = sharpen(gray)
        cv2.imwrite('output/watershed_laplacian.jpg', laplacian)
        cv2.imwrite('output/watershed_sharpened.jpg', sharpened)
        
        blurred = cv2.GaussianBlur(sharpened, self.gaussian_kernel, 0)
        cv2.imwrite('output/watershed_blurred.jpg', blurred)
        
        thresh = self.apply_threshold(blurred)
        cv2.imwrite('output/watershed_thresh.jpg', thresh)
        
        filled_image = self.apply_morphology(thresh)
        cv2.imwrite('output/watershed_filled_image.jpg', filled_image)
        
        overlap = input("Press 1 if there is significant overlap in image : ")
        if overlap == '1':
            filled_image = self.erode(filled_image)
            cv2.imwrite('output/watershed_eroded.jpg', filled_image)

        distance_transform,sure_fg = self.compute_distance_transform(filled_image)
        cv2.imwrite('output/watershed_distance_transform.jpg', distance_transform)
        cv2.imwrite('output/watershed_sure_fg.jpg', sure_fg)

        total_coins,watershed_result = self.apply_watershed(img, sure_fg, filled_image)
        cv2.imwrite('output/watershed_result.jpg', watershed_result)

        print("Total coins detected:", total_coins)
        return watershed_result


if __name__ == "__main__":
    
    path1 = "images/hough_input.jpg"
    cd = HoughCoinDetector()
    cd.process_hough(path1)

    path2 = "images/watershed_input.jpg"
    wt = WatershedProcessor()
    wt.process_watershed(path2)

