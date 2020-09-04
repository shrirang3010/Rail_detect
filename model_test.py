from model import vgg_unet 
import cv2
import numpy as np

vidcap = cv2.VideoCapture('\\rail\\InputVideo.mov')
success,image = vidcap.read()
count = 0
success = True

model = vgg_unet(n_classes=3,  input_height=416, input_width=608  )

model.load_weights('vgg_unet_1.3')
fourcc = cv2.VideoWriter_fourcc(*"XVID")
outvideo = cv2.VideoWriter('\\rail\\videoout\\OutputVideo.mp4',fourcc , 14, (3840,2160))

while vidcap.isOpened():
  #cv2.imwrite("frame%d.png" % count, image)   
  success,image = vidcap.read()
  count += 1
  out = model.predict_segmentation(
      inp=image,
      #inp= vidcap,
      out_fname=None
  )
  if count == 750:
    break
  out = out.astype(np.uint8)
  new_img = cv2.addWeighted(image, 1, out, 0.7, 0, image)
  cv2.imwrite(f'\\rail\\videoout\\{count}.png', new_img)
  outvideo.write(new_img)

outvideo.release()
vidcap.release()
exit()
