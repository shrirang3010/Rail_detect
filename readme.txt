Segmentation of Rail-tracks using Keras(Deep Learning)

-- Model Used: VGG_unet

-- Input: video:	/rail/InputVideo
	  train:         /rail/frames
	  annotation:	/rail/new_png_masks
	  model:	vgg_unet_1.3	
   The original png_masks had to be compressed in size

-- Output: video:	/rail/videoout/OutputVideo

VGG_unet -- Proposed by Oxford has lesser layers thus making it faster to train. VGG has been the standard pre-trained model in for a large number of applications.
VGG being an innovative object-recognition model supporting up to 19 layers helped in building a network which recognised/segmented the rail tracks quiet accurately.

To run the code:
-- Copy the 'Codes' folder in your environment
-- Open terminal in the folder
-- Check if all import modules are installed
-- Run model_train.py to train the model (-- Check the paths for input and output file as required)
-- Run model_test.py to get the desired segnented output as both images and video