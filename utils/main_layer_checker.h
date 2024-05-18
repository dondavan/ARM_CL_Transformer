#ifndef __MAIN_LAYER_CHECKER__
#define __MAIN_LAYER_CHECKER__

#include <set>
#include <map>
#include <string>
#include <unordered_set>



inline std::map<std::string, std::unordered_set<std::string>> ending_task_names{
	{
		"alexnet",
		{
			"pool1",
			"pool2",
			"relu3",
			"relu4",
			"pool5",
			"relu6",
			"relu7",
			"prob"
		}
	},
	{
		"googlenet",
		{
			"pool1/norm1",
			"pool2/3x3_s2",
			"inception_3a/concat",
			"pool3/3x3_s2",
			"inception_4a/concat",
			"inception_4b/concat",
			"inception_4c/concat",
			"inception_4d/concat",
			"pool4/3x3_s2",
			"inception_5a/concat",
			"prob"
		}
	},
	{
		"mobilenetv1",
		{
			"Conv2d_0/Relu6",
			"Conv2d_1_pointwise/Relu6",
			"Conv2d_2_pointwise/Relu6",
			"Conv2d_3_pointwise/Relu6",
			"Conv2d_4_pointwise/Relu6",
			"Conv2d_5_pointwise/Relu6",
			"Conv2d_6_pointwise/Relu6",
			"Conv2d_7_pointwise/Relu6",
			"Conv2d_8_pointwise/Relu6",
			"Conv2d_9_pointwise/Relu6",
			"Conv2d_10_pointwise/Relu6",
			"Conv2d_11_pointwise/Relu6",
			"Conv2d_12_pointwise/Relu6",
			//"Logits/AvgPool_1a",//This layer is after "Conv2d_13_pointwise/Relu6"
			"Softmax",
			"concat",
		}
	},
	/*{
		"mobilenetv1",
		{
			"Conv2d_0/Relu6",
			//"Conv2d_1_depthwise/Relu6",
			"Conv2d_2_depthwise/Relu6",
			"Conv2d_3_depthwise/Relu6",
			"Conv2d_4_depthwise/Relu6",
			"Conv2d_5_depthwise/Relu6",
			"Conv2d_6_depthwise/Relu6",
			"Conv2d_7_depthwise/Relu6",
			"Conv2d_8_depthwise/Relu6",
			"Conv2d_9_depthwise/Relu6",
			"Conv2d_10_depthwise/Relu6",
			"Conv2d_11_depthwise/Relu6",
			"Conv2d_12_depthwise/Relu6",
			"Conv2d_13_depthwise/Relu6",
			"Logits/AvgPool_1a",
			"Softmax"
		}
	},*/
	{
		"resnetv1_50",
		{
			"pool1/MaxPool",
			"block1/unit1/bottleneck_v1/add",
			"block1/unit2/bottleneck_v1/add",
			"block1/unit3/bottleneck_v1/add",
			"block2/unit1/bottleneck_v1/add",
			"block2/unit2/bottleneck_v1/add",
			"block2/unit3/bottleneck_v1/add",
			"block2/unit4/bottleneck_v1/add",
			"block3/unit1/bottleneck_v1/add",
			"block3/unit2/bottleneck_v1/add",
			"block3/unit3/bottleneck_v1/add",
			"block3/unit4/bottleneck_v1/add",
			"block3/unit5/bottleneck_v1/add",
			"block3/unit6/bottleneck_v1/add",
			"block4/unit1/bottleneck_v1/add",
			"block4/unit2/bottleneck_v1/add",
			"pool5",
			"predictions/Softmax"
		}
	},
	{
		"squeezenetv1",
		{
			"pool1",
			"fire2/concat",
			"fire3/concat",
			"pool4",
			"fire5/concat",
			"fire6/concat",
			"fire7/concat",
			"pool8",
			"fire9/concat",
			"prob"
		}
	},

	{
		"yolov3",
		{
			"conv2d_1/LeakyRelu",
			"conv2d_2/LeakyRelu",
			"conv2d_3/LeakyRelu",
			"add_3_4",
			"conv2d_5/LeakyRelu",
			"conv2d_6/LeakyRelu",
			"add_6_7",
			"conv2d_8/LeakyRelu",
			"add_8_9",
			"conv2d_10/LeakyRelu",
			"conv2d_11/LeakyRelu",
			"add_11_12",
			"conv2d_13/LeakyRelu",
			"add_13_14",
			"conv2d_15/LeakyRelu",
			"add_15_16",
			"conv2d_17/LeakyRelu",
			"add_17_18",
			"conv2d_19/LeakyRelu",
			"add_19_20",
			"conv2d_21/LeakyRelu",
			"add_21_22",
			"conv2d_23/LeakyRelu",
			"add_23_24",
			"conv2d_25/LeakyRelu",
			"add_25_26",
			"conv2d_27/LeakyRelu",
			"conv2d_28/LeakyRelu",
			"add_28_29",
			"conv2d_30/LeakyRelu",
			"add_30_31",
			"conv2d_32/LeakyRelu",
			"add_32_33",
			"conv2d_34/LeakyRelu",
			"add_34_35",
			"conv2d_36/LeakyRelu",
			"add_36_37",
			"conv2d_38/LeakyRelu",
			"add_38_39",
			"conv2d_40/LeakyRelu",
			"add_40_41",
			"conv2d_42/LeakyRelu",
			"add_42_43",
			"conv2d_44/LeakyRelu",
			"conv2d_45/LeakyRelu",
			"add_45_46",
			"conv2d_47/LeakyRelu",
			"add_47_48",
			"conv2d_49/LeakyRelu",
			"add_49_50",
			"conv2d_51/LeakyRelu",
			"add_51_52",
			"conv2d_53/LeakyRelu",
			"conv2d_54/LeakyRelu",
			"conv2d_55/LeakyRelu",
			"conv2d_56/LeakyRelu",
			"conv2d_57/LeakyRelu",
			"conv2d_58/LeakyRelu",
			"Yolo1",
			//"conv2d_60/LeakyRelu",
			//"Upsample_60",
			"Route1",
			"conv2d_61/LeakyRelu",
			"conv2d_62/LeakyRelu",
			"conv2d_63/LeakyRelu",
			"conv2d_64/LeakyRelu",
			"conv2d_65/LeakyRelu",
			"conv2d_66/LeakyRelu",
			"Yolo2",
			//"conv2d_68/LeakyRelu",
			//"Upsample_68",
			"Route2",
			"conv2d_69/LeakyRelu",
			"conv2d_70/LeakyRelu",
			"conv2d_71/LeakyRelu",
			"conv2d_72/LeakyRelu",
			"conv2d_73/LeakyRelu",
			"conv2d_74/LeakyRelu",
			"Yolo3" 
		}
	}



	
	
};


inline std::map<std::string, std::unordered_set<std::string>> starting_task_names{
	{
		"alexnet",
		{ "conv1",
			//"conv2_g0",
			//"conv2_g1",
			"conv2",
			"conv3",
			//"conv4_g0",
			//"conv4_g1",
			"conv4",
			//"conv5_g0",
			//"conv5_g1",
			"conv5",
			"fc6",
			"fc7",
			"fc8",
		}
	},
	{
		"googlenet",
		{ "conv1/7x7_s2",
			"conv2/3x3_reduce",
			"inception_3a/1x1",
			"inception_3b/1x1",
			"inception_4a/1x1",
			"inception_4b/1x1",
			"inception_4c/1x1",
			"inception_4d/1x1",
			"inception_4e/1x1",
			"inception_5a/1x1",
			"inception_5b/1x1",
		}
	},
	{
		"mobilenetv1",
		{ "Conv2d_0",
			"Conv2d_1_depthwise/depthwise",
			"Conv2d_2_depthwise/depthwise",
			"Conv2d_3_depthwise/depthwise",
			"Conv2d_4_depthwise/depthwise",
			"Conv2d_5_depthwise/depthwise",
			"Conv2d_6_depthwise/depthwise",
			"Conv2d_7_depthwise/depthwise",
			"Conv2d_8_depthwise/depthwise",
			"Conv2d_9_depthwise/depthwise",
			"Conv2d_10_depthwise/depthwise",
			"Conv2d_11_depthwise/depthwise",
			"Conv2d_12_depthwise/depthwise",
			"Conv2d_13_depthwise/depthwise",
		}
	},

	{
		"resnetv1_50",
		{
			"conv1/convolution",
			"block1/unit1/bottleneck_v1/conv1/convolution",
			"block1/unit2/bottleneck_v1/conv1/convolution",
			"block1/unit3/bottleneck_v1/conv1/convolution",
			"block2/unit1/bottleneck_v1/conv1/convolution",
			"block2/unit2/bottleneck_v1/conv1/convolution",
			"block2/unit3/bottleneck_v1/conv1/convolution",
			"block2/unit4/bottleneck_v1/conv1/convolution",
			"block3/unit1/bottleneck_v1/conv1/convolution",
			"block3/unit2/bottleneck_v1/conv1/convolution",
			"block3/unit3/bottleneck_v1/conv1/convolution",
			"block3/unit4/bottleneck_v1/conv1/convolution",
			"block3/unit5/bottleneck_v1/conv1/convolution",
			"block3/unit6/bottleneck_v1/conv1/convolution",
			"block4/unit1/bottleneck_v1/conv1/convolution",
			"block4/unit2/bottleneck_v1/conv1/convolution",
			"block4/unit3/bottleneck_v1/conv1/convolution",
			"logits/convolution",
		}

	},
	{
		"squeezenetv1",
		{
			"conv1",
			"fire2/squeeze1x1",
			"fire3/squeeze1x1",
			"fire4/squeeze1x1",
			"fire5/squeeze1x1",
			"fire6/squeeze1x1",
			"fire7/squeeze1x1",
			"fire8/squeeze1x1",
			"fire9/squeeze1x1",
			"conv10",
		}
	},

	{
		"yolov3",
		{
			"conv2d_1/Conv2D",
			"conv2d_2/Conv2D",
			"conv2d_3/Conv2D",
			"conv2d_4/Conv2D",
			"conv2d_5/Conv2D",
			"conv2d_6/Conv2D",
			"conv2d_7/Conv2D",
			"conv2d_8/Conv2D",
			"conv2d_9/Conv2D",
			"conv2d_10/Conv2D",
			"conv2d_11/Conv2D",
			"conv2d_12/Conv2D",
			"conv2d_13/Conv2D",
			"conv2d_14/Conv2D",
			"conv2d_15/Conv2D",
			"conv2d_16/Conv2D",
			"conv2d_17/Conv2D",
			"conv2d_18/Conv2D",
			"conv2d_19/Conv2D",
			"conv2d_20/Conv2D",
			"conv2d_21/Conv2D",
			"conv2d_22/Conv2D",
			"conv2d_23/Conv2D",
			"conv2d_24/Conv2D",
			"conv2d_25/Conv2D",
			"conv2d_26/Conv2D",
			"conv2d_27/Conv2D",
			"conv2d_28/Conv2D",
			"conv2d_29/Conv2D",
			"conv2d_30/Conv2D",
			"conv2d_31/Conv2D",
			"conv2d_32/Conv2D",
			"conv2d_33/Conv2D",
			"conv2d_34/Conv2D",
			"conv2d_35/Conv2D",
			"conv2d_36/Conv2D",
			"conv2d_37/Conv2D",
			"conv2d_38/Conv2D",
			"conv2d_39/Conv2D",
			"conv2d_40/Conv2D",
			"conv2d_41/Conv2D",
			"conv2d_42/Conv2D",
			"conv2d_43/Conv2D",
			"conv2d_44/Conv2D",
			"conv2d_45/Conv2D",
			"conv2d_46/Conv2D",
			"conv2d_47/Conv2D",
			"conv2d_48/Conv2D",
			"conv2d_49/Conv2D",
			"conv2d_50/Conv2D",
			"conv2d_51/Conv2D",
			"conv2d_52/Conv2D",
			"conv2d_53",
			"conv2d_54",
			"conv2d_55",
			"conv2d_56",
			"conv2d_57",
			"conv2d_58",
			"conv2d_59",
			"conv2d_60",
			"conv2d_61",
			"conv2d_62",
			"conv2d_63",
			"conv2d_64",
			"conv2d_65",
			"conv2d_66",
			"conv2d_67",
			"conv2d_68",
			"conv2d_69",
			"conv2d_70",
			"conv2d_71",
			"conv2d_72",
			"conv2d_73",
			"conv2d_74",
			"conv2d_75"
		}
	}





};
	
inline std::string toLowerCase(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return str;
}

inline bool check_ending(std::string model, std::string layer){	
	if (layer.compare(0, 4, "NPU_") == 0){
		return true;
	}
	return (ending_task_names[toLowerCase(model)].count(layer) > 0);
}

inline bool check_starting(std::string model, std::string layer){
	if (layer.compare(0, 4, "NPU_") == 0){
		return true;
	}
	return (starting_task_names[toLowerCase(model)].count(layer) > 0);
}

	


#endif /* __UTILS_UTILS_H__*/
