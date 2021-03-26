## evaluationmouvementsgmmPython
This is a Python3 version of 'EvaluationMouventsgmm'. The purpose is to allow the Poppy to perform mouvment learning and evaluation autonomously. Make sure to put the data files in the same directory. 

To start the program, use the commands below. 

```
python mainLearning.py
python mainEvaluation.py
```
Make sure that you have Python3 installed on your computer. If 'python' is not the corresponding environment variable for Python3, replace it with your environment variable for Python3.

'MainLearning.py' should create a file named 'model.txt', it stores the data of our trained GMM model in serialized format, it's normal that you can't read it directely. It will then be read in 'mainEvaluation.py'.


The kinect skeleton data files are in folder data (.txt files). Each Kinect file corresponds to a motion sequence. Within the file, each row represents a frame and includes 175 floating values corresponding to the 3D position and the orientation (Quaternion) of each joint:
x_pos y_pos z_pos x_quat y_quat z_quat w_quat

Each joint's position and quaternion is concatenated  to include 25 joints:
0. SpineBase
1. SpineMid
2. Neck
3. Head
4. ShoulderLeft
5. ElbowLeft
6. WristLeft
7. HandLeft
8. ShoulderRight
9. ElbowRight
10. WristRight
11. HandRight
12. HipLeft
13. KneeLeft
14. AnkleLeft
15. FootLeft
16. HipRight
17. KneeRight
18. AnkleRight
19. FootRight
20. SpineShoulder
21. HandTipLeft
22. ThumbLeft
23. HandTipRight
24. ThumbRight


