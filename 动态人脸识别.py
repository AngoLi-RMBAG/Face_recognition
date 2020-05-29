'''
从摄像头捕获实时画面进行人脸检测

'''
import cv2
# 准备面部检测引擎和人眼检测引擎
# 注意调用方法没有任何区别
# 只是选择的xml文件不一样
# 面部检测是haarcascade_frontalface_default.xml
# 人眼检测是haarcascade_eye.xml
face_engine = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# 调用VideoCapture函数
# 参数设为0表示打开本地默认摄像头
# 参数若为文件路径也可以打开一个本地的视频文件
# 返回一个VideoCapture对象
cap = cv2.VideoCapture(0)
# 应该是设置屏幕亮度的参数
# 默认以下设置
cap.set(10,10)
# 由于人脸检测是在静态的图片上进行
# 所以需要对摄像头捕获的内容进行逐帧检测
# 定义下面的循环体
while True:
    # 调用read函数读取每一帧的图片
    # 返回两个值
    # success表示若图片读取成功返回True
    # frame表示返回的Image对象
    success, frame = cap.read()
    # 使用面部识别引擎检测人脸
    # 返回一个包含面部区域信息的可迭代对象
    faces = face_engine.detectMultiScale(frame, 1.3, 3)
    # 对每个面部区域画出矩形
    for (x,y,w,h) in faces:
        # 在图片frame上绘制矩形
        img1 = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # 截取人脸区域
        # 因为人眼在人脸上
        # 所以人眼检测一定在面部区域进行
        # 可以提高检测的效率
        face_area = img1[y:y+h,x:x+w]
        # 人眼检测
        eyes = eye_cascade.detectMultiScale(face_area,1.3,3)
        # 绘制人眼矩形区域
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # 绘制好矩形区域后，创建图片显示窗口
    cv2.imshow('Face_recognition', frame)
    # 条件限制
    # 表示键盘每隔5毫秒监听一次
    # 0xFF == ord('q')表示按下键盘上的Q键触发动作break退出循环
    # 注意这里的Q键必须要在英文输入法条件下进行
    if cv2.waitKey(5) & 0xFF == ord('q'):
            break
# 调用release函数关闭摄像头
# 这个要特别注意!
cap.release()
# 顾名思义，清理窗口、退出窗口
cv2.destroyAllWindows()







