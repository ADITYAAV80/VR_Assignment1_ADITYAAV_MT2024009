import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import read_image

class ImageStitcher:
    def __init__(self, focal_length=1100):
        self.focal_length = focal_length

    def find_matches(self, base_image, sec_image):
        """
        Finds keypoints and matches between two images using SIFT and Brute Force Matcher.
        """
        sift = cv2.SIFT_create()
        base_kp, base_des = sift.detectAndCompute(cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY), None)
        sec_kp, sec_des = sift.detectAndCompute(cv2.cvtColor(sec_image, cv2.COLOR_BGR2GRAY), None)

        cv2.imwrite("output/panaroma_BaseImage_kp.jpg", cv2.drawKeypoints(base_image, base_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
        cv2.imwrite("output/panaroma_SecImage_kp.jpg", cv2.drawKeypoints(sec_image, sec_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = sorted(bf.match(base_des, sec_des), key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * 0.75)]

        cv2.imwrite("output/panaroma_Good_matches.jpg", cv2.drawMatches(base_image, base_kp, sec_image, sec_kp, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

        return good_matches, base_kp, sec_kp

    def find_homography(self, matches, base_kp, sec_kp):
        """
        Computes the homography matrix using RANSAC.
        """
        if len(matches) < 4:
            print("\nNot enough matches found between the images.\n")
            exit(0)

        base_pts = np.float32([base_kp[m.queryIdx].pt for m in matches])
        sec_pts = np.float32([sec_kp[m.trainIdx].pt for m in matches])
        homography_matrix, _ = cv2.findHomography(sec_pts, base_pts, cv2.RANSAC, 5.0)

        return homography_matrix

    def get_new_frame_size_and_matrix(self, homography_matrix, sec_shape, base_shape):
        """
        Calculates the new frame size after transformation.
        """
        h, w = sec_shape
        corners = np.array([[0, 0, 1], [w-1, 0, 1], [w-1, h-1, 1], [0, h-1, 1]]).T
        transformed_corners = np.dot(homography_matrix, corners)
        transformed_corners /= transformed_corners[2]
        x, y = transformed_corners[:2]

        min_x, min_y = np.floor(x.min()).astype(int), np.floor(y.min()).astype(int)
        max_x, max_y = np.ceil(x.max()).astype(int), np.ceil(y.max()).astype(int)

        correction_x, correction_y = max(0, -min_x), max(0, -min_y)
        new_width = max_x + correction_x
        new_height = max_y + correction_y

        new_width = max(new_width, base_shape[1] + correction_x)
        new_height = max(new_height, base_shape[0] + correction_y)

        offset_matrix = np.array([[1, 0, correction_x], [0, 1, correction_y], [0, 0, 1]])
        homography_matrix = np.dot(offset_matrix, homography_matrix)

        return [new_height, new_width], (correction_x, correction_y), homography_matrix

    def cylindrical_projection(self, img):
        """
        Applies cylindrical projection to an image.
        """
        h, w = img.shape[:2]
        K = np.array([[self.focal_length, 0, w//2], [0, self.focal_length, h//2], [0, 0, 1]])
        map_x, map_y = np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        mask_x, mask_y = [], []
        for y in range(h):
            for x in range(w):
                X, Y = (x - w//2) / self.focal_length, (y - h//2) / self.focal_length
                Z = np.sqrt(X**2 + 1)
                new_x, new_y = self.focal_length * np.arctan(X) + w//2, self.focal_length * Y / Z + h//2

                if 0 <= new_x < w and 0 <= new_y < h:
                    map_x[y, x], map_y[y, x] = new_x, new_y
                    mask_x.append(int(new_x))
                    mask_y.append(int(new_y))

        transformed_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        return transformed_img, np.array(mask_x), np.array(mask_y)

    def stitch_images(self,BaseImage, SecImage):
        
        # Applying Cylindrical projection on SecImage
        SecImage_Cyl, mask_x, mask_y = self.cylindrical_projection(SecImage)

        # Getting SecImage Mask
        SecImage_Mask = np.zeros(SecImage_Cyl.shape, dtype=np.uint8)
        SecImage_Mask[mask_y, mask_x, :] = 255

        # Finding matches between the 2 images and their keypoints
        Matches, BaseImage_kp, SecImage_kp = self.find_matches(BaseImage, SecImage_Cyl)

        # Finding homography matrix.
        HomographyMatrix = self.find_homography(Matches, BaseImage_kp, SecImage_kp)

        # Finding size of new frame of stitched images and updating the homography matrix
        NewFrameSize, Correction, HomographyMatrix = self.get_new_frame_size_and_matrix(HomographyMatrix, SecImage_Cyl.shape[:2], BaseImage.shape[:2])

        # Transform images to the new coordinate frame
        SecImage_Transformed = cv2.warpPerspective(SecImage_Cyl, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
        SecImage_Transformed_Mask = cv2.warpPerspective(SecImage_Mask, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))

        #     # Ensure the mask is smooth (float values between 0 and 1)
        # SecImage_Transformed_Mask = cv2.GaussianBlur(SecImage_Transformed_Mask, (21, 21), sigmaX=10, sigmaY=10)

        # # Convert mask to 3 channels for blending (if images are colored)
        # SecImage_Transformed_Mask = cv2.merge([SecImage_Transformed_Mask] * 3)

        # # Blend images smoothly instead of bitwise operations

        
        # # plt.imshow(cv2.cvtColor(SecImage_Transformed,cv2.COLOR_BGR2RGB))
        # # plt.show()
        # # plt.imshow(cv2.cvtColor(SecImage_Transformed_Mask,cv2.COLOR_BGR2RGB))
        # # plt.show()
        # BaseImage_Transformed = np.zeros((NewFrameSize[0], NewFrameSize[1], 3), dtype=np.uint8)
        # BaseImage_Transformed[Correction[1]:Correction[1]+BaseImage.shape[0], Correction[0]:Correction[0]+BaseImage.shape[1]] = BaseImage

        # _, BinaryMask = cv2.threshold(SecImage_Transformed_Mask, 1, 255, cv2.THRESH_BINARY)

        # # Define center for blending
        # center = (NewFrameSize[1] // 2, NewFrameSize[0] // 2)

        # StitchedImage = (SecImage_Transformed * SecImage_Transformed_Mask) + (BaseImage_Transformed * (1 - SecImage_Transformed_Mask))
        # StitchedImage = StitchedImage.astype(np.uint8)

        #StitchedImage = cv2.bitwise_or(SecImage_Transformed, cv2.bitwise_and(BaseImage_Transformed, cv2.bitwise_not(SecImage_Transformed_Mask)))

        # # Apply Poisson blending
        # #StitchedImage = feather_blend(SecImage_Transformed, BaseImage_Transformed, BinaryMask)

        # return StitchedImage

        #SecImage_Transformed = cv2.warpPerspective(SecImage_Cyl, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
        #SecImage_Transformed_Mask = cv2.warpPerspective(SecImage_Mask, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))

        # Smoothen the mask
        SecImage_Transformed_Mask = cv2.GaussianBlur(SecImage_Transformed_Mask, (3, 3), sigmaX=0, sigmaY=0)

        # Normalize the mask (important!)
        SecImage_Transformed_Mask = SecImage_Transformed_Mask.astype(np.float32) / 255.0

        # Convert to 3 channels if necessary
        #if len(SecImage_Transformed_Mask.shape) == 2:
        #    SecImage_Transformed_Mask = cv2.merge([SecImage_Transformed_Mask] * 3)

        # Initialize the base image with black background
        BaseImage_Transformed = np.zeros((NewFrameSize[0], NewFrameSize[1], 3), dtype=np.uint8)
        BaseImage_Transformed[Correction[1]:Correction[1]+BaseImage.shape[0], Correction[0]:Correction[0]+BaseImage.shape[1]] = BaseImage

        # Blending images smoothly using weighted sum
        alpha = SecImage_Transformed_Mask
        beta = 1 - alpha
        StitchedImage = (SecImage_Transformed * alpha) + (BaseImage_Transformed * beta)
        StitchedImage = np.clip(StitchedImage, 0, 255).astype(np.uint8)

        #     # Convert stitched image to grayscale
        # gray = cv2.cvtColor(StitchedImage, cv2.COLOR_BGR2GRAY)

        # # Create mask for dark regions (thresholding)
        # _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)  # Pixels close to black

        # # Inpaint black regions using surrounding pixels
        # StitchedImage = cv2.inpaint(StitchedImage, mask, inpaintRadius=20, flags=cv2.INPAINT_TELEA)


        #StitchedImage = cv2.GaussianBlur(StitchedImage, (7, 7), sigmaX=0, sigmaY=0)


        return StitchedImage

if __name__ == "__main__":

    images = []
    for i in range(1,4):
        images.append(read_image(f"input/img{i}.png"))

    stitcher = ImageStitcher()
    base_image, _, _ = stitcher.cylindrical_projection(images[0])

    for i in range(1, len(images)):
        base_image = stitcher.stitch_images(base_image, images[i])

    plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
    plt.show()