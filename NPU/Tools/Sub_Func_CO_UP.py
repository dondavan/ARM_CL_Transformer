# +
#from ast import In
#from posixpath import split
from pyexpat import model
#from zlib import DEF_MEM_LEVEL
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb
#import google.protobuf.text_format
#from caffe import layers as L
import caffe
import os
import keras
from keras.utils.vis_utils import plot_model
import re

import convert_co_up
import test_data

import warnings
warnings.filterwarnings("ignore")
import time

#import tensorflow as tf



# +
Main_Layers={}
Main_Layers['bvlc_alexnet.caffemodel']=[#'data',
										'conv1',
										'conv2',
										'conv3',
										'conv4',
										'conv5',
										'fc6',
										'fc7',
										'fc8']

Main_Layers['bvlc_googlenet.caffemodel']=[#'data',
										  'conv1/7x7_s2',
										  'conv2/3x3_reduce',
										  #'conv2/3x3',
										  'inception_3a/1x1',
										  'inception_3b/1x1',
										  'inception_4a/1x1',
										  'inception_4b/1x1',
										  'inception_4c/1x1',
										  'inception_4d/1x1',
										  'inception_4e/1x1',
										  'inception_5a/1x1',
										  #'inception_5b/1x1',
										  'loss3/classifier']


# layers start with depthwise
Main_Layers['MobileNet.h5']=[#'input_2',
							 'conv1_pad',
							 'conv_dw_1',
							 #'conv_pw_1',
							 'conv_pad_2',
							 #'conv_pw_2',
							 'conv_dw_3',
							 #'conv_pw_3',
							 'conv_pad_4',
							 #'conv_pw_4',
							 'conv_dw_5',
							 #'conv_pw_5',
							 'conv_pad_6',
							 #'conv_pw_6',
							 'conv_dw_7',
							 #'conv_pw_7',
							 'conv_dw_8',
							 #'conv_pw_8',
							 'conv_dw_9',
							 #'conv_pw_9',
							 'conv_dw_10',
							 #'conv_pw_10',
							 'conv_dw_11',
							 #'conv_pw_11',
							 'conv_pad_12',
							 #'conv_pw_12',
							 'conv_dw_13',
							 #'conv_pw_13',
							 #'conv_preds'
							]

# layers start by pointwise (In partial quantiztion dataset gathering I gether data with this method 
# so here and in armcl also split before pointwise )
'''Main_Layers['MobileNet.h5']=[#'input_2',
							 'conv1_pad',
							 #'conv_dw_1',
							 'conv_pw_1',
							 #'conv_pad_2',
							 'conv_pw_2',
							 #'conv_dw_3',
							 'conv_pw_3',
							 #'conv_pad_4',
							 'conv_pw_4',
							 #'conv_dw_5',
							 'conv_pw_5',
							 #'conv_pad_6',
							 'conv_pw_6',
							 #'conv_dw_7',
							 'conv_pw_7',
							 #'conv_dw_8',
							 'conv_pw_8',
							 #'conv_dw_9',
							 'conv_pw_9',
							 #'conv_dw_10',
							 'conv_pw_10',
							 #'conv_dw_11',
							 'conv_pw_11',
							 #'conv_pad_12',
							 'conv_pw_12',
							 #'conv_dw_13',
							 'conv_pw_13',
							 'conv_preds']
'''
Main_Layers['ResNet50.h5']=[#'input_1',
							'conv1_pad',
							'res2a_branch2a',
							'res2b_branch2a',
							'res2c_branch2a',
							'res3a_branch2a',
							'res3b_branch2a',
							'res3c_branch2a',
							'res3d_branch2a',
							'res4a_branch2a',
							'res4b_branch2a',
							'res4c_branch2a',
							'res4d_branch2a',
							'res4e_branch2a',
							'res4f_branch2a',
							'res5a_branch2a',
							'res5b_branch2a',
							'res5c_branch2a',
							'fc1000']

Main_Layers['squeezenet_v1.0.caffemodel']=[#'data',
										   'conv1',
										   'fire2/squeeze1x1',
										   #'fire2/expand1x1',
										   'fire3/squeeze1x1',
										   #'fire3/expand1x1',
										   'fire4/squeeze1x1',
										   #'fire4/expand1x1',
										   'fire5/squeeze1x1',
										   #'fire5/expand1x1',
										   'fire6/squeeze1x1',
										   #'fire6/expand1x1',
										   'fire7/squeeze1x1',
										   #'fire7/expand1x1',
										   'fire8/squeeze1x1',
										   #'fire8/expand1x1',
										   'fire9/squeeze1x1',
										   #'fire9/expand1x1',
										   'conv10','prob']

sorted_layers={}
sorted_layers['YOLOv3']=['input_2', 'conv_0', 'bnorm_0', 'leaky_0', 'zero_padding2d_1', 
			   'conv_1', 'bnorm_1', 'leaky_1', 
			   'conv_2', 'bnorm_2', 'leaky_2', 
			   'conv_3', 'bnorm_3', 'leaky_3', 'add_1', 'zero_padding2d_2', 
			   'conv_5', 'bnorm_5', 'leaky_5', 
			   'conv_6', 'bnorm_6', 'leaky_6', 
			   'conv_7', 'bnorm_7', 'leaky_7', 'add_2', 
			   'conv_9', 'bnorm_9', 'leaky_9', 
			   'conv_10', 'bnorm_10', 'leaky_10', 'add_3', 'zero_padding2d_3', 
			   'conv_12', 'bnorm_12', 'leaky_12', 
			   'conv_13', 'bnorm_13', 'leaky_13', 
			   'conv_14', 'bnorm_14', 'leaky_14', 'add_4', 
			   'conv_16', 'bnorm_16', 'leaky_16', 
			   'conv_17', 'bnorm_17', 'leaky_17', 'add_5', 
			   'conv_19', 'bnorm_19', 'leaky_19', 
			   'conv_20', 'bnorm_20', 'leaky_20', 'add_6', 
			   'conv_22', 'bnorm_22', 'leaky_22', 
			   'conv_23', 'bnorm_23', 'leaky_23', 'add_7', 
			   'conv_25', 'bnorm_25', 'leaky_25', 
			   'conv_26', 'bnorm_26', 'leaky_26', 'add_8', 
			   'conv_28', 'bnorm_28', 'leaky_28', 
			   'conv_29', 'bnorm_29', 'leaky_29', 'add_9', 
			   'conv_31', 'bnorm_31', 'leaky_31', 
			   'conv_32', 'bnorm_32', 'leaky_32', 'add_10', 
			   'conv_34', 'bnorm_34', 'leaky_34', 
			   'conv_35', 'bnorm_35', 'leaky_35', 'add_11', 'zero_padding2d_4', 
			   'conv_37', 'bnorm_37', 'leaky_37', 
			   'conv_38', 'bnorm_38', 'leaky_38', 
			   'conv_39', 'bnorm_39', 'leaky_39', 'add_12', 
			   'conv_41', 'bnorm_41', 'leaky_41', 
			   'conv_42', 'bnorm_42', 'leaky_42', 'add_13', 
			   'conv_44', 'bnorm_44', 'leaky_44', 
			   'conv_45', 'bnorm_45', 'leaky_45', 'add_14', 
			   'conv_47', 'bnorm_47', 'leaky_47', 
			   'conv_48', 'bnorm_48', 'leaky_48', 'add_15', 
			   'conv_50', 'bnorm_50', 'leaky_50', 
			   'conv_51', 'bnorm_51', 'leaky_51', 'add_16', 
			   'conv_53', 'bnorm_53', 'leaky_53', 
			   'conv_54', 'bnorm_54', 'leaky_54', 'add_17', 
			   'conv_56', 'bnorm_56', 'leaky_56', 
			   'conv_57', 'bnorm_57', 'leaky_57', 'add_18', 
			   'conv_59', 'bnorm_59', 'leaky_59', 
			   'conv_60', 'bnorm_60', 'leaky_60', 'add_19', 'zero_padding2d_5', 
			   'conv_62', 'bnorm_62', 'leaky_62', 
			   'conv_63', 'bnorm_63', 'leaky_63', 
			   'conv_64', 'bnorm_64', 'leaky_64', 'add_20', 
			   'conv_66', 'bnorm_66', 'leaky_66', 
			   'conv_67', 'bnorm_67', 'leaky_67', 'add_21', 
			   'conv_69', 'bnorm_69', 'leaky_69', 
			   'conv_70', 'bnorm_70', 'leaky_70', 'add_22', 
			   'conv_72', 'bnorm_72', 'leaky_72', 
			   'conv_73', 'bnorm_73', 'leaky_73', 'add_23', 
			   'conv_75', 'bnorm_75', 'leaky_75', 
			   'conv_76', 'bnorm_76', 'leaky_76', 
			   'conv_77', 'bnorm_77', 'leaky_77', 
			   'conv_78', 'bnorm_78', 'leaky_78', 
			   'conv_79', 'bnorm_79', 'leaky_79', 
			   'conv_80', 'bnorm_80', 'leaky_80', 
			   'conv_81', 
			   'conv_84', 'bnorm_84', 'leaky_84', 'up_sampling2d_1', 'concatenate_1', 
			   'conv_87', 'bnorm_87', 'leaky_87', 
			   'conv_88', 'bnorm_88', 'leaky_88', 
			   'conv_89', 'bnorm_89', 'leaky_89', 
			   'conv_90', 'bnorm_90', 'leaky_90', 
			   'conv_91', 'bnorm_91', 'leaky_91', 
			   'conv_92', 'bnorm_92', 'leaky_92', 
			   'conv_93', 
			   'conv_96', 'bnorm_96', 'leaky_96', 'up_sampling2d_2', 'concatenate_2', 
			   'conv_99', 'bnorm_99', 'leaky_99', 
			   'conv_100', 'bnorm_100', 'leaky_100', 
			   'conv_101', 'bnorm_101', 'leaky_101', 
			   'conv_102', 'bnorm_102', 'leaky_102', 
			   'conv_103', 'bnorm_103', 'leaky_103', 
			   'conv_104', 'bnorm_104', 'leaky_104', 
			   'conv_105']
# +
granularity="Conv"

if granularity=="Conv":
	formatPattern = r"^(?!.*_g\d*)(?!.*relu)(?!.*batchnorm)(?!.*linear)(.*conv.*|.*fc.*)"
	# If you do not want to skip input and output layers:
	# formatPattern = r"^(?!.*_g\d*)(?!.*relu)(?!.*batchnorm)(?!.*linear)(.*conv.*|.*fc.*|$)"

#
if granularity=="Operation":
	formatPattern = r"^(?!.*_g\d*)(?!.*relu)(?!$)"

index = 0

def check_name(name):
	global index
	if re.search(formatPattern, name, re.IGNORECASE):
		index += 1
		print(f"{index} layer: {name}")
		return True
	else:
		print(f"{index} skipping layer: {name}")
		return False


# -

#_dir="/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/Sub_Model/"
_dir="/home/ehsan/UvA/Sub_Model/"

def set_main_layers(Model_name):
	global Main_Layers
	layers=[l.name for l in net.layer]
	#sorted_layers=sort_strings_by_number(layers)
	#Main_Layers['Yolov3.h5']=[l for l in sorted_layers if 'padding' not in l and 'bias' not in l]
	##Main_Layers[Model_name]=[l for l in layers if 'relu' not in l and 'drop' not in l]
	Main_Layers[Model_name]=[l for l in layers if check_name(l)]
	print(f"main layers of the model {Model_name} are {Main_Layers[Model_name]}")
	print(len(Main_Layers[Model_name]))

'''def set_main_layers_keras(Model_name):
	global Main_Layers
	layers=[l.name for l in model.layers]
	#sorted_layers=sort_strings_by_number(layers)
	#Main_Layers['Yolov3.h5']=[l for l in sorted_layers if 'padding' not in l and 'bias' not in l]
	#Main_Layers[Model_name]=[l for l in layers if 'relu' not in l and 'drop' not in l]
	Main_Layers[Model_name]=[l for l in layers if check_name(l)]
	print(f"main layers of the model {Model_name} are {Main_Layers[Model_name]}")
	print(len(Main_Layers[Model_name]))'''

# +
def extract_number(string):
	# Extract the number from the string
	pattern = r"(\d+)"
	match = re.search(pattern, string)
	if match:
		return int(match.group(0))
	return 0

def sort_strings_by_number(strings):
	# Sort the strings based on the embedded numbers
	sorted_strings = sorted(strings, key=extract_number)
	return sorted_strings



# +
def get_outbound_layers(layer):
	outbound_layers=[]
	if(isinstance(layer, str)):
		layer=model.get_layer(layer_name)
	for node in layer._outbound_nodes:
		outbound_layer=node.outbound_layer if hasattr(node, 'outbound_layer') else None
		outbound_layers.append(outbound_layer)
	return outbound_layers
		
def get_inbound_layers(layer):
	inbound_layers=[]
	if(isinstance(layer, str)):
		layer=model.get_layer(layer)
	for node in layer._inbound_nodes:
		if hasattr(node, 'outbound_layers'):
			inbound_layer=node.inbound_layers 
			inbound_layers.append(inbound_layer)
		else:
			for inbound_tensor in node.input_tensors:
				if hasattr(inbound_tensor, '_keras_history'):
					#inbound_layer = inbound_tensor._keras_history.layer
					inbound_layer = inbound_tensor._keras_history[0]
					inbound_layers.append(inbound_layer)
	return inbound_layers

def traverse_layers(layer, visited_layers=set(), _sorted_layers=[]):
	# If the layer has already been visited, return
	if layer in visited_layers:
		return _sorted_layers
	visited_layers.add(layer)
	_sorted_layers.append(layer)

	# Process the current layer
	#print(f"Visiting layer: {layer.name}")

	# If the layer has branches (multiple outbound nodes), traverse each
	if hasattr(layer, '_outbound_nodes') and len(layer._outbound_nodes) > 1:
		both_conv=True
		for node in layer._outbound_nodes:
			if not isinstance(node.outbound_layer,keras.layers.Conv2D):
				both_conv=False
		if both_conv:
			for node in layer._outbound_nodes[::-1]:
				#for outbound_layer in node.outbound_layers:
				traverse_layers(node.outbound_layer, visited_layers, _sorted_layers)
		else:
			for node in layer._outbound_nodes:
				#for outbound_layer in node.outbound_layers:
				traverse_layers(node.outbound_layer, visited_layers, _sorted_layers)
	else:
		# If no branches, just go to the next layer
		if hasattr(layer, '_outbound_nodes') and len(layer._outbound_nodes) == 1:
			next_layer = layer._outbound_nodes[0].outbound_layer
			if isinstance(next_layer, list):
				# Some layers return a list even for single outbound layers
				next_layer = next_layer[0]
			traverse_layers(next_layer, visited_layers, _sorted_layers)

	return _sorted_layers


def set_main_layers_yolov3():
	global Main_Layers
	#modelfile='Yolo/Yolov3.h5'
	#model=keras.models.load_model(modelfile)
	#model.summary()
	layers=[l.name for l in model.layers]
	global sorted_layers
	#This does not work: (some layers lik add concat zero padding wil place in wrong order)
	#sorted_layers=sort_strings_by_number(layers)
	#insted this work:
	sorted_layers_1= traverse_layers( model.layers[0])
	sorted_layers['YOLOv3']=[l.name for l in sorted_layers_1]
	print("\n\n\n\n\n\n")
	print(sorted_layers)
	print("\n\n\n\n\n")
	#Main_Layers['Yolov3.h5']=[l for l in sorted_layers if 'padding' not in l and 'bias' not in l]
	Main_Layers['Yolov3.h5']=[l for l in sorted_layers['YOLOv3'] if 'conv' in l and 'bias' not in l]
	for i,layer in enumerate(Main_Layers['Yolov3.h5']):
		#print(layer)
		for l in get_inbound_layers(layer):
			#print(f'input layer is {l.name}')
			if "zero_padding" in l.name:
				Main_Layers['Yolov3.h5'][i]=l.name
				#print(f'main layer changed to {l.name}')
			#print("explored\n")
	#print(len(Main_Layers['Yolov3.h5']))
	print(Main_Layers['Yolov3.h5'])
	
'''set_main_layers_yolov3()
m=main_keras('Yolo/Yolov3.h5',5,15)
m.summary()
'''

# +
# Depreacated instead I use the get_inbound_layers and get_outbound_layers functions that are much better
input_layers_dict={}
output_layers_dict={}

def get_input_layers_keras(layer):
	_input_layers = []
	for node in layer._inbound_nodes:
		input_layers.extend([tensor._keras_history[0] for tensor in node.input_tensors])
	print(_input_layers)
	return _input_layers


def set_input_layers_dict_keras():
	for layer in model.layers:
		input_layers_dict[layer.name]=[]
		for node in layer._inbound_nodes:
			input_layers_dict[layer.name].extend([tensor._keras_history[0].name for tensor in node.input_tensors])







def set_output_layers_dict_keras():
	for layer in model.layers:
		output_layers_dict[layer.name]=[]
		for l in input_layers_dict:
			if layer.name in input_layers_dict[l]:
				output_layers_dict[layer.name].append(l)
				
				
				

# -

def Load_Net(M='Alex/bvlc_alexnet.caffemodel',Structure='Alex/deploy.prototxt'):
	global net 
	net = caffe_pb2.NetParameter()
	global Model  
	Model = caffe.Net(Structure, 1, weights=M)
	with open(Structure, 'r') as f:
		pb.text_format.Merge(f.read(), net)
	#set_main_layers(M.split('/')[-1])
	global main_layers
	main_layers=Main_Layers[M.split('/')[-1]]
	print(f'Model {M} loaded.')
	
	return net

def Load_Net_Keras(Model_name):
	global model
	model=keras.models.load_model(Model_name)
	#set_main_layers_keras(Model_name.split('/')[-1])
	print(Model_name)
	#model.summary()
	sorted_layers_1= traverse_layers( model.layers[0])
	sorted_layers[Model_name.split('/')[-1]]=[l.name for l in sorted_layers_1]

	if Model_name.split('/')[-1]=="Yolov3.h5":
		set_main_layers_yolov3()
	global main_layers
	main_layers=Main_Layers[Model_name.split('/')[-1]]
	print(f'Model {Model_name} loaded.')
	
	# setting the name of input layers and output layers for all layers (Depreacated!)
	#set_input_layers_dict_keras()               
	#set_output_layers_dict_keras()
	
	return model


'''Load_Net_Keras("Mobile/MobileNet.h5")
n=[l.name for l in model.layers]
n'''


def Save_Net(Name):
	#Name=Name+'.prototxt'
	with open(Name, 'w') as f:
		f.write(pb.text_format.MessageToString(net))

	print(f'Model saved as {Name}')


def Fill_Indexes():
	global dict
	dict={}
	layers=net.layer
	print(len(layers))
	layer=0
	started=0
	print(main_layers)
	for i in range(len(layers)):
		if layers[i].name in main_layers:
			if started:
				dict[layer].setdefault('end',i-1)
				print(layer,dict[layer]['name'],dict[layer]['start'],dict[layer]['end'])
				layer=layer+1
				dict.setdefault(layer,{})
				dict[layer].setdefault('name',main_layers[layer])
				dict[layer].setdefault('start',i)
			else:                               
				dict.setdefault(layer,{})
				dict[layer].setdefault('name',main_layers[layer])
				dict[layer].setdefault('start',i)
				started=1
		if i==(len(layers)-1):
			dict[layer].setdefault('end',i)
			print(layer,dict[layer]['name'],dict[layer]['start'],dict[layer]['end'])


# +
def Fill_Indexes_keras_0(model_name):
	global dict
	dict={}
	#net=keras.models.load_model(model_name)
	layers=model.layers
	print(len(layers))
	layer=0
	started=0
	for i in range(len(layers)):
		#print(layers[i].name)
		
		if layers[i].name in main_layers:
			if started:
				dict[layer].setdefault('end',i-1)
				print(layer,dict[layer]['name'],dict[layer]['start'],dict[layer]['end'])
				layer=layer+1
				dict.setdefault(layer,{})
				dict[layer].setdefault('name',main_layers[layer])
				dict[layer].setdefault('start',i)
				print(f'\n\n\nlayer {main_layers[layer]}, input :{layers[i].input.shape}')
			else:                               
				dict.setdefault(layer,{})
				dict[layer].setdefault('name',main_layers[layer])
				dict[layer].setdefault('start',i)
				started=1
				print(f'\n\n\nlayer {main_layers[layer]}, input :{layers[i].name},{layers[i].input.shape}')
				
		print(f'sublayer {layers[i].name}, output shape :{layers[i].output.shape}')
		
		if i==(len(layers)-1):
			dict[layer].setdefault('end',i)
			print(layer,dict[layer]['name'],dict[layer]['start'],dict[layer]['end'])

		
# -

# +
def Fill_Indexes_keras(model_name):
	global dict
	dict={}
	#net=keras.models.load_model(model_name)
	layers=sorted_layers[model_name]
	print(len(layers))
	layer=0
	started=0
	for i in range(len(layers)):
		#print(layers[i].name)
		
		if layers[i] in main_layers:
			if started:
				dict[layer].setdefault('end',i-1)
				print(layer,dict[layer]['name'],dict[layer]['start'],dict[layer]['end'])
				layer=layer+1
				dict.setdefault(layer,{})
				dict[layer].setdefault('name',main_layers[layer])
				dict[layer].setdefault('start',i)
				print(f'\n\n\nlayer {main_layers[layer]}, input :{model.get_layer(layers[i]).input.shape}')
			else:                               
				dict.setdefault(layer,{})
				dict[layer].setdefault('name',main_layers[layer])
				dict[layer].setdefault('start',i)
				started=1
				print(f'\n\n\nlayer {main_layers[layer]}, input :{layers[i]},{model.get_layer(layers[i]).input.shape}')
				
		print(f'sublayer {layers[i]}, output shape :{model.get_layer(layers[i]).output.shape}')
		
		if i==(len(layers)-1):
			dict[layer].setdefault('end',i)
			print(layer,dict[layer]['name'],dict[layer]['start'],dict[layer]['end'])




def Slice(Start,End):
	# Extract Input shape of start layer
	global pb_convert_args
	pb_convert_args={}
	Bottom_Name=Model.bottom_names[main_layers[Start]][0]
	print(f'Previous layer name: {Bottom_Name}')
	Input_Shape=Model.blobs[Bottom_Name].data.shape
	if len(Input_Shape) == 2:
		Input_Shape=Input_Shape[:1]+(1,1,)+Input_Shape[1:]
	print(f'Input shape is: {Input_Shape}')
	pb_convert_args["input_shape"]=[]
	pb_convert_args["input_shape"].append(Input_Shape)
	#for b in Model.blobs:
	#	print(Model.blobs[b].data.shape)

	# Set input shape to Extracted Shape
	input=net.layer[0]
	shape=input.input_param.shape[0]
	shape.Clear()
	shape.dim.MergeFrom(Input_Shape)

	# Slice the model using indexed dict
	Start_index=dict[Start]['start']
	End_index=dict[End]['end']
	print(f'Start and end indexes are: {Start_index,End_index}')
	del net.layer[End_index+1:]
	previous_layers=[Bottom_Name]
	if Start_index>0:
		previous_layer_name=net.layer[Start_index-1].name
		previous_layers.append(previous_layer_name)
	if Start>0:     
		previous_layer_name2=main_layers[Start-1]
		previous_layers.append(previous_layer_name2)
	#print(f'start:{Start}, end:{End}, p:{previous_layer_name}, p2:{previous_layer_name2} ')
	del net.layer[1:Start_index]

	'''
	# Connect start layer to input layer 
	C1=net.layer[1]
	C1.ClearField('bottom')
	C1.bottom.append(input.name)
	'''

	# Connect start layers to input layer (Considering multiple parallel input layer)
	#print(f'Name of previousl layer {previous_layer_name} and {previous_layer_name2} and also {Bottom_Name}')
	print(f'Name of previous layers: {previous_layers}')
	for l in net.layer:
		print(f'bottom of {l.name}:{l.bottom}')
		#if l.bottom==[previous_layer_name] or l.bottom==[previous_layer_name2] or l.bottom==[Bottom_Name]:
		if l.bottom and l.bottom[0] in previous_layers:
			print(f'new first layer after data:{l}')
			l.ClearField('bottom')
			l.bottom.append(input.name)
	#print(net.layer)
	
	print(f'Model sliced')

def split_keras(model_name,Start,End):
	

	Start_index=dict[Start]['start']
	End_index=dict[End]['end']

	#model=keras.models.load_model(model_name)
	#DL_input = keras.layers.Input(model.layers[indx].input_shape[1:])
	print(f'Input shape is:{model.layers[Start_index].get_input_shape_at(0)[1:]}')
	print(f'Start and end indexes are: {Start_index,End_index}')
	
	p_layer=model.layers[Start_index-1]
	DL_input = keras.layers.Input(model.layers[Start_index].get_input_shape_at(0)[1:],name='my_input')
	DL_model = DL_input
	DL_model = model.layers[Start_index](DL_model)
	#ll=model.layers[:]
	for layer in model.layers[Start_index+1:End_index+1]:
		layer_in_shape=0
		if isinstance(layer.input, list):
			layer_in_shape=layer.input[0].shape
		else:
			layer_in_shape=layer.input.shape
		DL_model_name=0
		DL_model_shape=0
		if isinstance(DL_model,list):
			DL_model_shape=DL_model[0].shape
			DL_model_name=DL_model[0].name
		else:
			DL_model_shape=DL_model.shape
			DL_model_name=DL_model.name
		print(f'adding layer: {layer.name} with shape {layer_in_shape} to {DL_model_name} with shape {DL_model_shape}')
		if type(layer.input)==type([]): 
			if p_layer.output in layer.input:
				DL_model = layer([DL_input,DL_model])
			else:
				for l in model.layers:
					if l.output in layer.input:
						DL_model = layer([l.get_output_at(1),DL_model])
						break
		else:
			DL_model = layer(DL_model)
		
	DL_model = keras.models.Model(inputs=DL_input, outputs=DL_model)
	
	plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	print(f'Model sliced')
	_dir=os.path.dirname(model_name)
	_ext=os.path.splitext(model_name)[1]
	new_name=_dir+'/'+model_name.split('/')[-2]+'_'+str(Start)+'_'+str(End)+_ext
	pb_name=_dir+'/'+model_name.split('/')[-2]+'_'+str(Start)+'_'+str(End)+'.pb'
	DL_model.save(new_name)
	#DL_model.summary()
	print(f'Model saved as {new_name}')
	#print(f'Model input:{DL_model.input.name}, Output:{DL_model.output.name}, input_shape:{DL_model.input.shape}')
	global pb_convert_args
	pb_convert_args={}
	pb_convert_args['h5name']=new_name
	pb_convert_args['pb_name']=pb_name
	pb_convert_args['input']=DL_model.layers[0].name
	pb_convert_args['output']=DL_model.layers[-1].get_output_at(0).op.name
	shape=''
	s=list(DL_model.layers[0].get_input_shape_at(0)[1:])
	for x in s:
		shape=shape+str(x)+','
	shape=shape[:-1]
	pb_convert_args['input_shape']=shape
	for l in DL_model.layers:
		print(f'name:{l.name} output:{l.get_output_at(0).name}')

	return DL_model


def split_keras_2(model_name,Start,End):
	Start_index=dict[Start]['start']
	End_index=dict[End]['end']

	#model=keras.models.load_model(model_name)
	
	Input_shape=model.layers[Start_index].get_input_shape_at(0)[1:]
	'''if isinstance(model.layers[Start_index].get_input_shape_at(0), tuple):
		Input_shapes=[model.layers[Start_index].get_input_shape_at(0)[1:]]
	
	if isinstance(model.layers[Start_index].get_input_shape_at(0), list):
		Input_shapes=[shape[1:] for shape in model.layers[Start_index].get_input_shape_at(0)]'''
	
	print(f'Input shape is:{Input_shape}')
	print(f'Start and end indexes are: {Start_index,End_index}')
	p_layer=model.layers[Start_index-1]
	nodes=p_layer._outbound_nodes
	Input_layer=keras.layers.InputLayer(input_shape=Input_shape,name="New_input")
	for node in nodes:
		if p_layer.output in node.input_tensors:
			node.input_tensors.remove(p_layer.output)
		node.input_tensors.append(Input_layer.output)

		if p_layer in node.inbound_layers:
			node.inbound_layers.remove(p_layer)
		node.inbound_layers.append(Input_layer)
	Input_layer._outbound_nodes=nodes
	if 'dropout' in model.layers[End_index].name:
		print(f'The last layer is dropout, it should be deleted because error in converting to rknn (dropout should be in training no inference!)')
		End_index=End_index-1
	new_model=keras.models.Model(inputs=Input_layer.input,outputs=model.layers[End_index].output)
	'''if 'dropout' in new_model.layers[-1].name:
		print(f'The last layer is dropout, it should be deleted because error in converting to rknn (dropout should be in training no inference!)')
		new_model.layers.pop()'''
		
	print(f'\n\nNew model First Layer:{new_model.layers[1].name}, Input shape:{new_model.layers[0].input.shape}\n\
			Input name:{new_model.layers[0].name} First layer input shape:{new_model.layers[1].input.shape},\n\
			Last layer name:{new_model.layers[-1].name} and {new_model.layers[-1].get_output_at(0).op.name}\n\
			output shape:{new_model.layers[-1].output.shape}')

	plot_model(new_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	print(f'Model sliced')
	_dir=os.path.dirname(model_name)
	_ext=os.path.splitext(model_name)[1]
	new_name=_dir+'/'+model_name.split('/')[-2]+'_'+str(Start)+'_'+str(End)+_ext
	pb_name=_dir+'/'+model_name.split('/')[-2]+'_'+str(Start)+'_'+str(End)+'.pb'
	new_model.save(new_name)
	#DL_model.summary()
	print(f'Model saved as {new_name}')
	#print(f'Model input:{DL_model.input.name}, Output:{DL_model.output.name}, input_shape:{DL_model.input.shape}')
	global pb_convert_args
	pb_convert_args={}
	pb_convert_args['h5name']=new_name
	pb_convert_args['pb_name']=pb_name
	pb_convert_args['input']=new_model.layers[0].name
	pb_convert_args['output']=new_model.layers[-1].get_output_at(0).op.name
	shape=''
	s=list(new_model.layers[0].get_input_shape_at(0)[1:])
	for x in s:
		shape=shape+str(x)+','
	shape=shape[:-1]
	pb_convert_args['input_shape']=shape
	return new_model




def split_keras_co_up_0(model_name,Start,End):
	Start_index=dict[Start]['start']
	End_index=dict[End]['end']

	#model=keras.models.load_model(model_name)
	
	if False:
		Start=3
		End=10
		Start_index=dict[Start]['start']
		End_index=dict[End]['end']
		Start_index=14
	
	
	input_names=[]
	input_shapes=[]
	input_tensors=[]
	output_names=[]
	output_tensors=[]
	inbound_layers_outside=[]
	outbound_layers_outside=[]
	sub_model=model.layers[Start_index:End_index+1]
	print(f'\n\n****************\nSlicing model from layer: {sub_model[0].name} to {sub_model[-1].name}')
	## Get inbound and outbound layers outside the sub_model
	for i,layer in enumerate(sub_model):
		inbound_layers=get_inbound_layers(layer)
		for inbound_layer in inbound_layers:
			if inbound_layer not in sub_model:
				inbound_layers_outside.append(inbound_layer)

		outbound_layers=get_outbound_layers(layer)
		for outbound_layer in outbound_layers:
			if outbound_layer not in sub_model:
				outbound_layers_outside.append(outbound_layer)
				#Filling the output_tensors (if is for speciall case that layer is dropout which should be ignored then)
				if 'dropout' in layer.name:
					print(f'The last layer is dropout, it should be deleted because error in converting to rknn (dropout should be in training no inference!)')
					output_tensors.append(sub_model[i-1].output)
					output_names.append(sub_model[i-1].output.op.name)
				else:
					output_tensors.append(layer.output)
					output_names.append(layer.output.op.name)
				
	
	# in case that start_index = 0 
	if Start_index==0:
		input_names.append(sub_model[0].name)
		input_shapes.append(sub_model[0].output.shape.as_list()[1:])
		#input_shapes.append(sub_model[0].get_output_shape_at(0)[1:])
		input_tensors.append(sub_model[0].output)
	# or end_index = len(layers)
	if(End_index==len(model.layers)-1):
		output_names.append(sub_model[-1].output.op.name)
		output_tensors.append(sub_model[-1].output)
	
	print("\ninbound layers outside:")
	for layer in inbound_layers_outside:
		print(layer.name)

	print("\noutbound layers outside:")
	for layer in outbound_layers_outside:
		print(layer.name)
	 
	
	## Replace inbound_layers_outside with input layers
	for inbound_layer_outside in inbound_layers_outside:
		#Input layer for replacement with inbounding layer outside
		print(f'Create input layer for {inbound_layer_outside.name} with shape: {tuple(inbound_layer_outside.output.shape.as_list()[1:])}')
		input_name=inbound_layer_outside.name
		input_shape=inbound_layer_outside.output.shape.as_list()[1:]
		Input_layer=keras.layers.InputLayer(input_shape=tuple(input_shape),name=input_name)
		#print(f'Input layer outbound nodes:{Input_layer._outbound_nodes}')
		input_tensors.append(Input_layer.output)
		input_names.append(input_name)
		input_shapes.append(input_shape)
		# explore the output nodes of the inbound layer outside (see node as connection point)
		for node in inbound_layer_outside._outbound_nodes:
			if node.outbound_layer in sub_model:
				print(f'Replacing connection from {inbound_layer_outside.name} to {node.outbound_layer.name}')
				if inbound_layer_outside.output in node.input_tensors:
					node.input_tensors.remove(inbound_layer_outside.output)
				node.input_tensors.append(Input_layer.output)
				if inbound_layer_outside in node.inbound_layers:
					node.inbound_layers.remove(inbound_layer_outside)
				node.inbound_layers.append(Input_layer)
				Input_layer._outbound_nodes.append(node)


	## Replace outbound_layers_outside with output layers
	#for outbound_layer_outside in outbound_layers_outside:
	print("\ninbound layers of input tensors:")
	for input_tensor in input_tensors:
		print(input_tensor._keras_history[0].name)

	print("\ninbound layers of output tensors:")
	for output_tensor in output_tensors:
		print(output_tensor._keras_history[0].name)
		
	

	new_model=keras.models.Model(inputs=input_tensors,outputs=output_tensors)

	new_model.summary()

	
		
	print(f'\n\nNew model First Layer:{new_model.layers[1].name}, Input shape:{new_model.layers[0].input.shape}\n\
			Input name:{new_model.layers[0].name} First layer input shape:{new_model.layers[1].input.shape},\n\
			Last layer name:{new_model.layers[-1].name} and {new_model.layers[-1].get_output_at(0).op.name}\n\
			output shape:{new_model.layers[-1].output.shape}')

	plot_model(new_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	print(f'Model sliced')
	_dir=os.path.dirname(model_name)
	_ext=os.path.splitext(model_name)[1]
	new_name=_dir+'/'+model_name.split('/')[-2]+'_'+str(Start)+'_'+str(End)+_ext
	pb_name=_dir+'/'+model_name.split('/')[-2]+'_'+str(Start)+'_'+str(End)+'.pb'
	new_model.save(new_name)
	#DL_model.summary()
	print(f'Model saved as {new_name}')
	#print(f'Model input:{DL_model.input.name}, Output:{DL_model.output.name}, input_shape:{DL_model.input.shape}')
	global pb_convert_args
	pb_convert_args={}
	pb_convert_args['h5name']=new_name
	pb_convert_args['pb_name']=pb_name
	pb_convert_args['input']=input_names
	pb_convert_args['output']=output_names
	pb_convert_args['input_shape']=input_shapes
	return new_model
# -

def remove_duplicates(my_list):
    seen = set()
    unique_list = []
    for item in my_list:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list



def split_keras_co_up(model_name,Start,End):
	
	
	Start_index=dict[Start]['start']
	End_index=dict[End]['end']

	#model=keras.models.load_model(model_name)
	
	if False:
		Start=3
		End=10
		Start_index=dict[Start]['start']
		End_index=dict[End]['end']
		Start_index=14
	
	
	input_names=[]
	input_shapes=[]
	input_tensors=[]
	output_names=[]
	output_tensors=[]
	inbound_layers_outside=[]
	outbound_layers_outside=[]
	#sub_model=model.layers[Start_index:End_index+1]
	sub_model=[model.get_layer(name) for name in sorted_layers[model_name.split('/')[-1]][Start_index:End_index+1]]

	#extracted_model = tf.keras.Sequential(sub_model)
	#extracted_model=keras.models.Model(sub_model)
	#extracted_model=keras.models.Sequential(sub_model)
	#extracted_model.summary()

	print(f'\n\n****************\nSlicing model from layer: {sub_model[0].name} to {sub_model[-1].name}')
	## Get inbound and outbound layers outside the sub_model
	for i,layer in enumerate(sub_model):
		inbound_layers=get_inbound_layers(layer)
		for inbound_layer in inbound_layers:
			if inbound_layer not in sub_model:
				inbound_layers_outside.append(inbound_layer)

		outbound_layers=get_outbound_layers(layer)
		for outbound_layer in outbound_layers:
			if outbound_layer not in sub_model:
				outbound_layers_outside.append(outbound_layer)
				#Filling the output_tensors (if is for speciall case that layer is dropout which should be ignored then)
				if 'dropout' in layer.name:
					print(f'The last layer is dropout, it should be deleted because error in converting to rknn (dropout should be in training no inference!)')
					output_tensors.append(sub_model[i-1].output)
					output_names.append(sub_model[i-1].output.op.name)
				else:
					output_tensors.append(layer.output)
					output_names.append(layer.output.op.name)
				
	
	# in case that start_index = 0 
	if Start_index==0:
		input_names.append(sub_model[0].name)
		input_shapes.append(sub_model[0].output.shape.as_list()[1:])
		#input_shapes.append(sub_model[0].get_output_shape_at(0)[1:])
		input_tensors.append(sub_model[0].output)

	'''# or end_index = len(layers)
	if(End_index==len(model.layers)-1):
		output_names.append(sub_model[-1].output.op.name)
		output_tensors.append(sub_model[-1].output)'''
	# add outputs of the original model if the correspinding layers are in sub_model
	for tensor in model.outputs:
		# TensorFlow 2.x approach; adjust if using a different backend
		layer_name = tensor.name.split('/')[0]
		layer = model.get_layer(name=layer_name)
		if layer in sub_model:
			output_tensors.append(tensor)
			output_names.append(layer.output.op.name)
	#method2:
	if False:
		for output in model.output_names:
			layer=model.get_layer(output)
			if layer in sub_model:
				output_tensors.append(layer.output)
				output_names.append(layer.output.op.name)

	
	print("\ninbound layers outside:")
	for layer in inbound_layers_outside:
		print(layer.name)

	print("\noutbound layers outside:")
	for layer in outbound_layers_outside:
		print(layer.name)
	 
	
	## Replace inbound_layers_outside with input layers
	for inbound_layer_outside in inbound_layers_outside:
		#Input layer for replacement with inbounding layer outside
		print(f'Create input layer for {inbound_layer_outside.name} with shape: {tuple(inbound_layer_outside.output.shape.as_list()[1:])}')
		input_name=inbound_layer_outside.name
		input_shape=inbound_layer_outside.output.shape.as_list()[1:]
		Input_layer=keras.layers.InputLayer(input_shape=tuple(input_shape),name=input_name)
		#print(f'Input layer outbound nodes:{Input_layer._outbound_nodes}')
		input_tensors.append(Input_layer.output)
		input_names.append(input_name)
		input_shapes.append(input_shape)
		# explore the output nodes of the inbound layer outside (see node as connection point)
		for node in inbound_layer_outside._outbound_nodes:
			if node.outbound_layer in sub_model:
				print(f'Replacing connection from {inbound_layer_outside.name} to {node.outbound_layer.name}')
				if inbound_layer_outside.output in node.input_tensors:
					node.input_tensors.remove(inbound_layer_outside.output)
				node.input_tensors.append(Input_layer.output)
				if inbound_layer_outside in node.inbound_layers:
					node.inbound_layers.remove(inbound_layer_outside)
				node.inbound_layers.append(Input_layer)
				Input_layer._outbound_nodes.append(node)


	input_tensors=remove_duplicates(input_tensors)
	input_names=remove_duplicates(input_names)
	output_tensors=remove_duplicates(output_tensors)
	output_names=remove_duplicates(output_names)
	
	## Replace outbound_layers_outside with output layers
	#for outbound_layer_outside in outbound_layers_outside:
	print("\ninbound layers of input tensors:")
	for input_tensor in input_tensors:
		print(input_tensor._keras_history[0].name)

	print("\ninbound layers of output tensors:")
	for output_tensor in output_tensors:
		print(output_tensor._keras_history[0].name)
		
	

	new_model=keras.models.Model(inputs=input_tensors,outputs=output_tensors)

	new_model.summary()

	
		
	print(f'\n\nNew model First Layer:{new_model.layers[1].name}, Input shape:{new_model.layers[0].input.shape}\n\
			Input name:{new_model.layers[0].name} First layer input shape:{new_model.layers[1].input.shape},\n\
			Last layer name:{new_model.layers[-1].name} and {new_model.layers[-1].get_output_at(0).op.name}\n\
			output shape:{new_model.layers[-1].output.shape}')

	plot_model(new_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	print(f'Model sliced')
	_dir=os.path.dirname(model_name)
	_ext=os.path.splitext(model_name)[1]
	new_name=_dir+'/'+model_name.split('/')[-2]+'_'+str(Start)+'_'+str(End)+_ext
	pb_name=_dir+'/'+model_name.split('/')[-2]+'_'+str(Start)+'_'+str(End)+'.pb'
	new_model.save(new_name)
	#DL_model.summary()
	print(f'Model saved as {new_name}')
	#print(f'Model input:{DL_model.input.name}, Output:{DL_model.output.name}, input_shape:{DL_model.input.shape}')
	global pb_convert_args
	pb_convert_args={}
	pb_convert_args['h5name']=new_name
	pb_convert_args['pb_name']=pb_name
	pb_convert_args['input']=input_names
	pb_convert_args['output']=output_names
	pb_convert_args['input_shape']=input_shapes
	return new_model
# -

if False:
	M="Yolo/Yolov3.h5"
	Load_Net_Keras(M)
	Fill_Indexes_keras(M)
	#In yolov3 the first leaky that have branch output
	l=model.layers[7]
	print(l.name)
	print(l.input)
	print(l.output)
	print(l._outbound_nodes)
	print(l._inbound_nodes)

	#In Yolov3 the first add that has input 
	l=model.layers[14]
	print(l.name)
	print(l.input)
	print(l.output)
	print(l._outbound_nodes)
	print(l._inbound_nodes)
	# As you can see each node could have mulitple input_tensors but just one output tensor
	# also l.output prints just one tensor, but layer.input could print two tensor (for example for 7)
#model.layers[7].output.shape.as_list()[1:]

def main(Start,End, M, Structure):
	Load_Net(M,Structure)
	print('inja')
	
	Fill_Indexes()
	print(dict)
	Slice(Start,End)

	
	_dir=os.path.dirname(Structure)
	_ext=os.path.splitext(Structure)[1]
	global Name
	Name=_dir+'/'+Structure.split('/')[-2]+'_'+str(Start)+'_'+str(End)+_ext
	Save_Net(Name)

def main_keras(M,Start,End):
	Load_Net_Keras(M)
	Fill_Indexes_keras(M.split('/')[-1])
	#print(dict)
	m=split_keras_co_up(M,Start,End)
	return m


Test=0
if Test:
	main_keras("Yolo/Yolov3.h5",4,10)

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description='Slice a model')
	parser.add_argument('--Model', metavar='path', required=False,
						help='Model')
	parser.add_argument('--Structure', metavar='path', required=False,
						help='Structure of the model (prototxt)')
	parser.add_argument('--Start', metavar='number', required=True,
						help='Starting layer')
	parser.add_argument('--End', metavar='number', required=True,
						help='Ending layer')

	
	args = parser.parse_args()
	print(f'st is {args.Structure}')
	if args.Model.split('.')[-1]=='h5':
		main_keras(args.Model,int(args.Start),int(args.End))
		if pb:
			cmd1=f'python {_dir}keras_to_tensorflow.py --input_model={pb_convert_args["h5name"]} --output_model={pb_convert_args["pb_name"]}'
			print(f'command is: {cmd1}')
			ok=os.system(cmd1)
			print(f'Freezing graph return code is {ok} ')
			
			#cmd2=f'python {_dir}convert.py {pb_convert_args["pb_name"]} {pb_convert_args["input"]} {pb_convert_args["output"]} {pb_convert_args["input_shape"]}'
			#print(f'command is: {cmd2}')
			#ok=os.system(cmd2)
			sample_dir="/home/ehsan/UvA/Sub_Model/dataset.txt"
			'''if "yolo" in pb_convert_args["pb_name"]:
				sample_dir="/home/ehsan/UvA/Accuracy/Keras/YOLOV3/Dataset/sample/"
			if "mobile" in pb_convert_args["pb_name"]:
				sample_dir="/home/ehsan/UvA/Accuracy/Imagenet/sample/"'''
			test_data.generate_dataset(pb_convert_args["input_shape"])
				
			ok=convert_co_up.convert(model_name=pb_convert_args["pb_name"],
									inputs=pb_convert_args["input"],
									outputs=pb_convert_args["output"],
									inputs_shape=pb_convert_args["input_shape"],
									quantized=True,
									quantization_dataset=sample_dir)
			print(f'Convert to rknn return code is {ok} ')
		
			
	else:
		main(Start=int(args.Start), End=int(args.End), M=args.Model, Structure=args.Structure)


		test_data.generate_dataset(pb_convert_args["input_shape"])
		sample_dir="/home/ehsan/UvA/Sub_Model/dataset.txt"
		quantized=True
		q=""
		if quantized:
			q="--quantized"
		cmd=f'python {_dir}convert_co_up.py --Structure={Name} --Model={args.Model} {q} --dataset={sample_dir}'
		print(f'command is {cmd}')
		ok=os.system(cmd)
		
		'''ok=convert_co_up.convert(model_name=Name,
								 #blobs=args.Model,
								 blobs="/home/ehsan/UvA/Sub_Model/Alex/bvlc_alexnet_1.caffemodel",
									quantized=False,
									quantization_dataset=sample_dir)'''
		print(f'Convert caffe to rknn return code is {ok}')

# +


def extract():
	ls=[l.name for l in model.layers]
	sls=sort_strings_by_number(ls)
	conv_index=0
	batch_norm_index=0
	layers_with_batch=-1
	#ls=model.layers[240]+model.layers[243]+model.layers[249]
	#ls=model.layers[241]+model.layers[244]+model.layers[250]
	#ls=model.layers[242]+model.layers[245]+model.layers[251]
	for kk,ll in enumerate(sls):
		#print(ll)
		l=model.get_layer(ll)
		for w in l.weights:
			name=w.name
			name2=name
			name=name.partition(':')[0]
			name=name.replace('/','_')
			name=name.replace('bnorm','batch_normalization')
			name=name.replace('moving_variance','var')
			name=name.replace('moving_mean','mean')
			pattern = r"(\d+)"
			#name = re.sub(pattern, lambda match: str(int(match.group(0)) + 1), name)
			#with each conv layer we increase the conve index by one (given that it is not conv_bias)
			if name.find('conv')==0:
				if name.find('bias') < 0:
					conv_index=conv_index+1
				name = re.sub(pattern, lambda match: str(conv_index), name)
				
			# the batch normalization indexing get different form conv indexing from layer conv59 which has not 
			# batch normalization (also layer 67 and 75 do not have bn)
			# so for batch-normaliztion layers for each 4 sub-layer (mean, var, gamma, beta) we increase its indexing
			if name.find('batch_normalization')==0:
				layers_with_batch+=1
				if layers_with_batch%4==0:
					batch_norm_index+=1
				name = re.sub(pattern, lambda match: str(batch_norm_index), name)
				
				
			name=name.replace('kernel','w')
			name=name.replace('conv','conv2d')
			name=name.replace('bias','b')
			name="yolov3_model/"+name+'.npy'
			# Extract the weights from the layer
			layer_weights = l.get_weights()

			# Get the list of variable names
			variable_names = [var.name for var in l.weights]
			for _name, _weight in zip(variable_names, layer_weights):
				if _name==name2:
					np.save(os.path.join(args.dumpPath, name),_weight)
			
			print(name,name2,kk)
# -



def Temp():
	input=net.layer[0];
	shape=kk=input.input_param.shape[0];
	shape.Clear();
	shape.dim.MergeFrom([10,256,13,13]);

	del net.layer[1:9]
	del net.layer[5:]
	C1=net.layer[1]
	C1.ClearField('bottom')
	C1.bottom.append(input.name)


