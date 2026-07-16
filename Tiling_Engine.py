import cv2

IMAGE_PATH="" 


img = cv2.imread(IMAGE_PATH) # numpy array of the image
def generate_tiles(img,tile_h,tile_w,overlap=0):
    img_height=img.shape[0]
    img_width=img.shape[1]
    stride_h, stride_w = tile_h - overlap,tile_w - overlap # overlap in pixels
    tiles=[]

    y=0

    while y < img_height:
        x = 0
        while x < img_width:
            y_end, x_end = min(y+tile_h,img_height),min(x+tile_w,img_width) # for bounds
            tile = img[y:y_end, x:x_end]
            tiles.append({"tile": tile, "x":x, "y":y})
            if x_end == img_width:
                break
            x+=stride_w
        if y_end == img_height:
            break
        y += stride_h

    return tiles
