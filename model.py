import keras
from keras.models import *
from keras.layers import *
from train import train
from predict import predict, predict_multiple, evaluate

IMAGE_ORDERING = 'channels_last'
pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
MERGE_AXIS = -1


class _C:
    def _m(self): pass
MethodType = type(_C()._m)

def vgg_unet( n_classes , input_height=416, input_width=608 , encoder_level=3):

	model =  _unet( n_classes , get_vgg_encoder ,  input_height=input_height, input_width=input_width  )
	model.model_name = "vgg_unet"
	return model

def get_segmentation_model( input , output ):

	img_input = input
	o = output

	o_shape = Model(img_input , o ).output_shape
	i_shape = Model(img_input , o ).input_shape

	if IMAGE_ORDERING == 'channels_first':
		output_height = o_shape[2]
		output_width = o_shape[3]
		input_height = i_shape[2]
		input_width = i_shape[3]
		n_classes = o_shape[1]
		o = (Reshape((  -1  , output_height*output_width   )))(o)
		o = (Permute((2, 1)))(o)
	elif IMAGE_ORDERING == 'channels_last':
		output_height = o_shape[1]
		output_width = o_shape[2]
		input_height = i_shape[1]
		input_width = i_shape[2]
		n_classes = o_shape[3]
		o = (Reshape((   output_height*output_width , -1    )))(o)

	o = (Activation('softmax'))(o)
	model = Model( img_input , o )
	model.output_width = output_width
	model.output_height = output_height
	model.n_classes = n_classes
	model.input_height = input_height
	model.input_width = input_width
	model.model_name = ""

	model.train = MethodType( train , model )
	model.predict_segmentation = MethodType( predict , model )
	model.predict_multiple = MethodType( predict_multiple , model )
	model.evaluate_segmentation = MethodType( evaluate , model )


	return model 

def _unet( n_classes , encoder , l1_skip_conn=True,  input_height=416, input_width=608  ):

	img_input , levels = encoder( input_height=input_height ,  input_width=input_width )
	[f1 , f2 , f3 , f4 , f5 ] = levels 

	o = f4

	o = ( ZeroPadding2D( (1,1) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([ o ,f3],axis=MERGE_AXIS )  )
	o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
	o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([o,f2],axis=MERGE_AXIS ) )
	o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING ) )(o)
	o = ( BatchNormalization())(o)

	o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	
	if l1_skip_conn:
		o = ( concatenate([o,f1],axis=MERGE_AXIS ) )

	o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
	o = ( BatchNormalization())(o)

	o =  Conv2D( n_classes , (3, 3) , padding='same', data_format=IMAGE_ORDERING )( o )
	
	model = get_segmentation_model(img_input , o )


	return model

def get_vgg_encoder( input_height=224 ,  input_width=224 , pretrained='imagenet'):

	assert input_height%32 == 0
	assert input_width%32 == 0

	if IMAGE_ORDERING == 'channels_first':
		img_input = Input(shape=(3,input_height,input_width))
	elif IMAGE_ORDERING == 'channels_last':
		img_input = Input(shape=(input_height,input_width , 3 ))

	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
	f1 = x
	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
	f5 = x

	
	if pretrained == 'imagenet':
		VGG_Weights_path = keras.utils.get_file( pretrained_url.split("/")[-1] , pretrained_url  )
		Model(  img_input , x  ).load_weights(VGG_Weights_path)


	return img_input , [f1 , f2 , f3 , f4 , f5 ]    


