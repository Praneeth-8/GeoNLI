import cv2

IMAGE_PATH="" 


img = cv2.imread(IMAGE_PATH) # numpy array of the image
def generate_tiles(img,height,width):
    img_height=img.shape[0]
    img_width=img.shape[1]
    tiles=[]

    for i in range(img_height//height):
        for j in range(img_width//width):
            tiles.append((img[i*height:(i+1)*height,j*width:(j+1)*width,],(i,j)))
        
    return tiles
