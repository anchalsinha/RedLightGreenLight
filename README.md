# Red Light, Green Light
### Use with Raspberry Pi Zero W Video Stream
Run the following command on the Raspberry Pi prior to starting the game
```
gst-launch-1.0 -v v4l2src ! video/x-raw,width=640,height=480 ! videoconvert ! jpegenc ! rtpjpegpay ! udpsink host=xxx.xxx.x.xxx port=5200
```
Ensure that the following line is uncommented in `game.py`
```
self.videoStream = cv2.VideoCapture('udpsrc port=5200 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)JPEG, payload=(int)26" ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
```
### Use with Webcam
Ensure that the following line is commented in `game.py`
```
self.videoStream = cv2.VideoCapture('udpsrc port=5200 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)JPEG, payload=(int)26" ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
```
Ensure that the following line is uncommented in `game.py`
```
self.videoStream = cv2.VideoCapture(0)
```
### Starting Game
Run `run.py` from within the `src` folder
