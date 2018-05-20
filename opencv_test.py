import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('data/input/10_min_sample_fit_test/frames/194.3.png')
edges = cv2.Canny(img, 150, 200)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                        minLineLength=100,
                        maxLineGap=5)

target_lines = np.empty(img.shape, np.uint8)

print(len(lines))

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(target_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.imshow(edges, cmap='gray')
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])
plt.show()

plt.imshow(target_lines, cmap='gray')
plt.title('Line detected')
plt.xticks([])
plt.yticks([])
plt.show()

'''
lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('test.png', img)
'''