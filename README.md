For our Hackathon project, we really sought to use image classification to assess health risks and promote the wellfare of the community. To do this, we chose to create an ML model that classifies food in a picture, which can then be uploaded either to a nutritional information application which gives nutritional info, or "Attila's Health App" which gives advice on the health impacts of the food you eat.


NUTRITIONAL INFORMATION APPLICATION

Uses the model we built to query USDA database, displaying the nutritional information given a picture of food.


ATTILA HEALTH APP

One of the applications we created created to show off our model is the Attila Health App, which uses Google's Gemini Flash 2.0 model to assess the health benefits and risks of the food in a given image. It's very simply put together but we think it came out looking quite nice. 


ML TRAINING BREAKDOWN (MacroVision)

The thought process behind this project was initially to train an ML model to detect the calories on a plate of food given
a photo. However, we quickly realized that this would rather difficult, given the fact that detecting which types of food
were in an image was not as straightforward as we had previously imagined.

There are numerous resources online for classifying images of food, such as the Food-101 dataset which we trained the
classification model on. However, these methods are only designed to work when given an image of a single food, as when given an image of a plate of various different foods, the model would give erroneous output. 

This is when we realized we had to also implement a way to detect the different foods in an image, separating them into their own "images", and run a classification model on those individual images. For this, we chose to use the YOLOv8 object detection algorithm to identify these separate foods. 

However, we once again ran into difficulty as the model would only seem to identify very broad objects, such as "plate" and "table". We realized we had to finetune the model in order to get it focus primarily on food. To do this, we utilized a dataset posted to roboflow (a website for computer vision projects/datasets), which had hundreds of images of food highlighted in images and allowed for us to get far better results. Here is the link to the dataset: https://universe.roboflow.com/gp-bvrcn/gp-n7jdm/dataset/4/download/yolov8, we trained it over 100 epochs which took roughly 90 minutes total.

The fun part of the project came when it was time to create the classification model. Originally we tried to go an easy route, attempting to use classes in models like ResNet and ImageNet to classify the foods. However, these classifications weren't very accurate as these models are more general purpose. Therefore, we decided to train our own classification model using the Food-101 dataset, which we only ran 10 epochs but took us around 3 hours since there are over 10000 images and labels.

After an ample amount of debugging, we were finally able to use our detection and classification models to fairly accurately predict the types of foods that are in a given image. However, the results are confined to 101 types of foods, as that is the extent of the Food-101 dataset, so it will often times label any green vegetable as "edamame" or mashed potatoes as "risotto". This implementation is certainly not the most effective, efficient, or comprehensive - but given the time constraints and limited prior knowledge, we are fairly happy with the results. Since our goal was originally to track calories, the name "MacroVision" stuck with us and we decided to name our model that.