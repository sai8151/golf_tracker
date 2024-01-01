
# 1.install python venv and start it
```
python3 -m venv demoenv
source demoenv/bin/activate
```

# 2.Install dependencies

```
pip install wheel numpy scipy matplotlib scikit-image scikit-learn ipython
pip install opencv-contrib-python
```

# 4.Download the YOLOv3 Weights:
```
wget https://pjreddie.com/media/files/yolov3.weights
```

# 5.Run app

```
python tracking.py
```

# Output
![image](https://epidotic-masts.000webhostapp.com/imagex.png?raw=true)



# this worked for me
```
sudo pip3 uninstall numpy
sudo pip uninstall numpy
sudo pip uninstall opencv
sudo pip3 install opencv-contrib-python
sudo pip3 install numpy
```
