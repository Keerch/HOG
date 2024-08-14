from skimage import feature, io, color
import matplotlib.pyplot as plt

# Wczytanie obrazu
image = io.imread('test.jpg') #Use path to your file

# Konwersja obrazu do skali szarości
gray_image = color.rgb2gray(image)

# Obliczenie cech HOG
fd, hog_image = feature.hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, channel_axis=None)
# Wizualizacja cech HOG
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(gray_image, cmap='gray')
plt.title('Obraz oryginalny')
plt.axis('off')
plt.subplot(122)
plt.imshow(hog_image, cmap='gray')
plt.title('Obraz HOG')
plt.axis('off')
plt.show()
