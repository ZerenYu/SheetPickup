import cv2
import os
import numpy as np

class sheetDetection:
    def __init__(self):
        print("Constructor for sheetDetection was called")
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.captures_path= os.path.join(self.dir_path, "images/captures")
        self.images_path= os.path.join(self.dir_path, "images/opencv_images")
        self.detected_images_path= os.path.join(self.dir_path, "images/detected_images")

    def __del__(self):
        print("Destructor for sheetDetection was called")

    def find_ArUco(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(self.img1)
        tag_dic = {}
        for i in range(0,len(markerIds)):
            tag_dic[markerIds[i][0]] = markerCorners[i][0]
        return tag_dic


    def narrow_img(self):
        tag_dic = self.find_ArUco()
        left = int(min(tag_dic[5].min(axis = 0)[0], tag_dic[6].min(axis = 0)[0]))
        right = int(max(tag_dic[2].max(axis = 0)[0], tag_dic[4].max(axis = 0)[0]))
        top = int(min(tag_dic[6].min(axis = 0)[1], tag_dic[2].min(axis = 0)[1]))
        bottom = int(max(tag_dic[5].max(axis = 0)[1], tag_dic[4].max(axis = 0)[1]))
        self.shift = (left, right, top, bottom)
        self.img2 = self.img1[top:bottom, left:right]

    def filter_blue(self):
        filter_size = 7
        modyfid = np.zeros(self.img2.shape).astype(np.int32)
        idx = 1
        for i in range(-3, 4):
            for j in range(-3, 4):
                temp = np.roll(self.img2, i, axis = 0).astype(np.int32)
                temp = np.roll(temp, j, axis = 1).astype(np.int32)
                modyfid = temp + modyfid
                # modyfid2 = modyfid/idx
                # cv2.imwrite(os.path.join(self.images_path, "{}-img.jpg".format(idx)), modyfid2)
                idx += 1
        self.img2 = (modyfid/49).astype(np.int32)
        # h, w,_ = self.img2.shape
        # for i in range(0, h):
        #     for j in range(0, w):
        #         left = max(j - int(filter_size/2), 0)
        #         right = min(j + int(filter_size/2)+1, w)
        #         top = max(i - int(filter_size/2), 0)
        #         bottom = min(i +int(filter_size/2)+1, h)
        #         pat = self.img2[top:bottom, left:right]
        #         self.img2[i, j] = pat.sum(axis = 0).sum(axis = 0)/(pat.shape[0] * pat.shape[1])


        cv2.imwrite(os.path.join(self.images_path, "blur-img.jpg"), self.img2)

        b = self.img2[:,:,0].astype(np.int32)
        g = self.img2[:,:,1].astype(np.int32)
        r = self.img2[:,:,2].astype(np.int32)
        b_ratio = b/(r+g+b) 
        b_ratio[b_ratio < 0.4] = 0
        b_ratio[b_ratio >= 0.4] = 1
        mask = b_ratio
        self.img2 = self.img2[:,:,1] * mask
        self.img2[self.img2 != 0] = 255

        # filter_size = 7
        # h, w = self.img2.shape
        # for i in range(0, h):
        #     for j in range(0, w):
        #         left = max(j - int(filter_size/2), 0)
        #         right = min(j + int(filter_size/2)+1, w)
        #         top = max(i - int(filter_size/2), 0)
        #         bottom = min(i +int(filter_size/2)+1, h)
        #         pat = self.img2[top:bottom, left:right]
        #         self.img2[i, j] = pat.sum()/(pat.shape[0] * pat.shape[1])
        modyfid = np.zeros(self.img2.shape).astype(np.int32)
        idx = 1
        for i in range(-3, 4):
            for j in range(-3, 4):
                temp = np.roll(self.img2, i, axis = 0).astype(np.int32)
                temp = np.roll(temp, j, axis = 1).astype(np.int32)
                modyfid = temp + modyfid
                # modyfid2 = modyfid/idx
                # cv2.imwrite(os.path.join(self.images_path, "{}-img.jpg".format(idx)), modyfid2)
                idx += 1
        self.img2 = (modyfid/49).astype(np.int32)
        

        self.img2[self.img2 > 80] = 255
        self.img2[self.img2 <= 80] = 0
        return


    def load_image(self, img_file):
        print("load_image:")
        self.img_file=img_file
        self.img1= cv2.imread(os.path.join(self.captures_path, self.img_file))
        self.narrow_img()
        self.filter_blue()
        cv2.imwrite(os.path.join(self.images_path, "1-img.jpg"), self.img2)
        self.image_size=self.img1.shape
        self.line_thickness=int(self.image_size[0]/300)

    def find_corners(self):
        gray = np.float32(self.img2)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=9,qualityLevel=0.02, minDistance=20, blockSize=21)

        print("find_corners:")
        self.mycorners = np.asarray(corners[:, 0, :]).astype(np.int32)
        self.mycorners += np.array([self.shift[0], self.shift[2]])
        return

    def output_image(self):
        print("output_image:")
        for point in self.mycorners:
            self.img1= cv2.drawContours(self.img1,np.array([[point]]),-1,(0,0,255),2)
            self.img1= cv2.circle(self.img1,(point[0],point[1]),self.line_thickness*5,(0,0,255),self.line_thickness*2)
        cv2.imwrite(os.path.join(self.detected_images_path, "output.jpg"), self.img1)
        

def main(input_image):
        mysheet= sheetDetection()
        mysheet.load_image(input_image)
        mysheet.find_corners()
        mysheet.output_image()
        return mysheet.mycorners


if __name__ == '__main__':
        input_image="40.jpg"
        output_points=main(input_image)
        print(output_points)