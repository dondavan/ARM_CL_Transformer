#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <map>

std::map<std::string,std::string> Models{{"Alex","bvlc_alexnet.caffemodel"},
										 {"Google","bvlc_googlenet.caffemodel"},
										 {"SqueezeV1","squeezenet_v1.0.caffemodel"},
										 {"MobileV1","MobileNet.h5"},
										 {"ResV1_50","ResNet50.h5"},
										 {"YOLOv3","Yolov3.h5"}};
std::map<std::string,std::string> Structures{{"Alex","deploy.prototxt"},
											{"Google","deploy.prototxt"},
											{"SqueezeV1","deploy.prototxt"},
											{"MobileV1",""},
											{"ResV1_50",""},
											{"YOLOv3",""}};

//std::string _dir="/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/Sub_Model/";
std::string _dir="/home/ehsan/UvA/Sub_Model/";




void Slice_Model(std::string model, std::string structure, int start, int end){
	std::ostringstream command;
	command<<"conda run -n rock-kit3 python "<<_dir<<"Sub_Func_CO_UP.py --Start="<<start<<" --End="<<end<<" --Model="<<model<<" --Structure="<<structure;
	std::cout<<command.str()<<std::endl;
	system(command.str().c_str());

	
}

void Convert(std::string model, std::string structure){
	//command.str("");
	//command.clear();
	std::ostringstream command;
	command<<"conda run -n rock-kit3 python "<<_dir<<"convert.py "<<structure<<" "<<model;
	system(command.str().c_str());
}
int main(int argc, char *argv[]){
	//Model
	
	std::string CNN=argv[1];
	std::string CNN_b=CNN;
	if (!CNN.empty()) {
        CNN_b[0] = std::toupper(CNN[0]);
    }
	int start=std::stoi(argv[2]);
	int end=std::stoi(argv[3]);
	
	
	std::string Model=_dir+CNN+"/"+Models[CNN];
	std::string Structure=_dir+CNN+"/"+Structures[CNN];
	//Start and End layers for slicing
	

	std::string Sliced=_dir+CNN+"/"+CNN+"_"+std::to_string(start)+"_"+std::to_string(end)+".prototxt";
	Slice_Model(Model,Structure,start,end);
	//Convert(Model,Sliced);
	return 0;
}
