# Object_Detection
Object detection with tensorflow to detect whale tails

Code was addapted from several other repositories, but mainly they are files from tensorflow examples libraries.
The objective was to use inception.v2 to create a object detection model that could detect whale tails. 

## References and Requirements
Original files can be found at [Tensorflow Models](https://github.com/tensorflow/models), this library is also needed for running most of the python scripts in this project and for other resources.  

Dataset was extracted from [kaggle](https://www.kaggle.com/c/humpback-whale-identification)  

[Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)  
[Google Protobuf](https://github.com/protocolbuffers/protobuf)

Imports:  
tensorflow  
numpy  
lxml  
PIL  
matplotlib  
IPhyton  
moviepy  


## 1_create_tf_record.py  
Paths need to be defined according to your setup

**data_dir** -> Main directory to where this project is saved to, it should contain the directories 'annotations' and 'images' with it's respective files;  
**output_dir** -> directory to where the TFrecords will be saved, for this project we used the main directory;  
**label_map_path** -> Path to 'label_map.pbtxt', for this project we saved this file at the main directory so we just especified is name and extension;  

This script will collect your pictures(data_dir/image/&ast;.jpg), bounding boxes (data_dir/annotations/&ast;.xml), list of examples (data_dir/annotations/trainval.txt) and classes (data_dir/label_map.pbtxt) to convert them to tensors, split them into training data and evaluation data, and output two TFrecord files with your data(data_dir/items_train.record and data_dir/items_val.record).  

## 2_train.py  
Paths need to be defined according to your setup  

**train_dir** -> output directory to where checkpoints and summaries will be saved, for this project we are pointing to main directory/train;  
**pipeline_config_path** -> .config file for your selected model, for this project we are using an adaptation of the sample config file supplied by tensorflow models library, they can be found at TensorflowModelsLibrary/research/object_detection/samples/configs, paths must be configured at the config file you can find them at the end of the file.  

&ast; If the script can't access your records and model trought your config file, you should configure the bellow paths as well  

**train_config_path** -> path to items_train.record  
**input_config_path** -> path to items_val.record  
**model_config_path** -> path to model.ckpt (should be in the folder project directory/rcnn_inception_v2_coco/model.ckpt) 

This script will start your training and output sumarries and checkpoints that will them be converted to your inference graph

## 3_export_inference_graph.py  
Paths need to be defined according to your setup  

**pipeline_config_path** -> .config file for your selected model.  
**trained_checkpoint_prefix** -> Last checkpoint model for this project we used project directory/train/model.ckpt-500, the numerical value at the end should match the checkpoint you are using to build the inference graph.  
**output_directory** -> Path to where the script will output the new model, for this project we used project directory/IG  

## 4_eval.py 
Run in virtual enviorment terminal  
python eval.py  
    --logtostdeer  
    --checkpoint_dir= PathTo\train  
    --eval_dir= PathTo\test  
    --pipeline_config_path=‪ PathTo\IG\pipeline.config  
  

---
Testing scripts were based on the ones available at Priya Dwivedi [GIT](https://github.com/priya-dwivedi/Deep-Learning)

## 5_display_and_crop.py  
Uses images in the testing folder for displaying them with their detected bounding box and the confidence level of the model, it also output a cropped image with just the object of interrest.  

Detection output:  
![Imgur](https://i.imgur.com/8Gv1Vum.png)  

Cropped output:  
![Imgur](https://i.imgur.com/W6Atp0K.jpg)  

## 6_video_box.py
Uses a video snippet for breaking it down to images, detecting the object, drawing the boxes and confidence level and reasembling the snippet into another video.  

output example:  
[![Video output](http://img.youtube.com/vi/UNSm_3amiww/0.jpg)](https://youtu.be/UNSm_3amiww)
