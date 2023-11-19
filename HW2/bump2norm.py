import numpy as np
from PIL import Image
import cv2

def bumpmap2normalmap(bumpmap, strength):
    height, width = bumpmap.shape
    print(bumpmap.shape)
    max_x = width - 1
    max_y = height - 1

    bump_map = bumpmap.astype(np.float32) / 255.0
    # Initialize normal map
    normal_map = np.zeros((height, width, 3), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            # Get the normal for this pixel
            # using central differences
            # calculate the value for the x-axis of the normal
            du = bump_map[y, min(x + 1, max_x)] - bump_map[y, max(x - 1, 0)]
            # calculate the value for the y-axis of the normal
            dv = bump_map[min(y + 1, max_y), x] - bump_map[max(y - 1, 0), x]
            # set the normal of this pixel
            du = - du * strength
            dv = - dv * strength
            normal = np.array([du, dv, 1.0])
            normal /= np.linalg.norm(normal)
            normal = (normal * 0.5 ) + 0.5
            normal_map[y, x] = normal

    res = normal_map * 255
    res = res.astype(np.uint8)
    #show,to bgr
    # cv2.imshow('normal', cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
    #save
    cv2.imwrite('earthnormal.jpg', cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
    return

bump = Image.open('earthbump.jpg')
bump = np.array(bump)
bumpmap2normalmap(bump, 3.5)