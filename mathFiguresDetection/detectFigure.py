from test.detectLabels import detectAlphabets
from removeLabels import removeLabels
import os

if __name__ == '__main__':
    input_image = os.path.join('test/images', 'Med_Ms_p0165_F_1_rgb_extracted.jpg')
    recognized, image = detectAlphabets(input_image)
    # imageNoLabel = removeLabels(recognized, image)
    #todo detect edges