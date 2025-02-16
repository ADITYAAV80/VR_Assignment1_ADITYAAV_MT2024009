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

    def stitch_images(self, base_image, sec_image):
        """
        Stitches two images together using cylindrical projection and homography.
        """
        sec_image_cyl, mask_x, mask_y = self.cylindrical_projection(sec_image)
        sec_mask = np.zeros(sec_image_cyl.shape, dtype=np.uint8)
        sec_mask[mask_y, mask_x, :] = 255

        matches, base_kp, sec_kp = self.find_matches(base_image, sec_image_cyl)
        homography_matrix = self.find_homography(matches, base_kp, sec_kp)
        new_size, correction, homography_matrix = self.get_new_frame_size_and_matrix(homography_matrix, sec_image_cyl.shape[:2], base_image.shape[:2])

        sec_transformed = cv2.warpPerspective(sec_image_cyl, homography_matrix, (new_size[1], new_size[0]))
        sec_mask_transformed = cv2.warpPerspective(sec_mask, homography_matrix, (new_size[1], new_size[0]))

        base_transformed = np.zeros((new_size[0], new_size[1], 3), dtype=np.uint8)
        base_transformed[correction[1]:correction[1]+base_image.shape[0], correction[0]:correction[0]+base_image.shape[1]] = base_image

        stitched_image = cv2.bitwise_or(sec_transformed, cv2.bitwise_and(base_transformed, cv2.bitwise_not(sec_mask_transformed)))

        cv2.imwrite("output/panaroma_stitchedImage.jpg", stitched_image)
        return stitched_image

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