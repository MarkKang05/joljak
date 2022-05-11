import cv2
img = cv2.imread("image/mm.png")
winname = "Test"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
cv2.imshow(winname, img)
cv2.waitKey()
cv2.destroyAllWindows()