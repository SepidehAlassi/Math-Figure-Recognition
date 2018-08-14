import matplotlib.pyplot as plt

def removeLabels(recognizedLabels, image):
    meanValue = image.mean()
    for label in recognizedLabels.keys():
        region = recognizedLabels[label]
        image[region[0]:region[2], region[1]:region[3]] = meanValue
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(image)
    plt.show()
    return image