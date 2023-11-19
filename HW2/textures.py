import numpy as np

def checkborad_pattern():
    img = np.zeros((512, 512, 3), 'uint8')
    for i in range(512):
        for j in range(512):
            c = (0.5 if ((i & 0x8 == 0) ^ (j & 0x8 == 0)) else 1) * 255
            img[i][j][0] = c
            img[i][j][1] = c
            img[i][j][2] = c
    
    return img