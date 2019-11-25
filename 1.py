import cv2

img = cv2.imread('./red_1.jpg')

cv2.imshow('demo', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
