from model import vgg_unet 

model = vgg_unet(n_classes=3 ,  input_height=416, input_width=608  )

model.train(
    train_images =  "\\rail\\frames\\",
    train_annotations = "\\rail\\new_png_masks\\",
    checkpoints_path = "\\rail\\vgg_unet_1" , epochs=5
)

