'''
　　安装OpenCV工具库
　　在Windows cmd(Windows 命令行)中输入下面命令：
　　pip install opencv_python -i https://pypi.tuna.tsinghua.edu.cn/simple
　　我一般喜欢在https://pypi.tuna.tsinghua.edu.cn/simple（清华开源软件镜像网站）
　　下载并安装第三方库，这样会大大提高安装的成功率，我比较推荐使用这种方式。
'''

# 导入cv2工具库，是的，opencv的导入名称是cv2而不是opencv
import cv2
# 读取一张本地图片：调用cv2中的imread函数，参数为图片文件的路径，返回一个Image对象
img = cv2.imread('Lena.jpg')
# 准备人脸识别引擎：这个部分我们暂且不聊，你就当作是调用一个实现人脸识别的应用程序了
# 注意调用的格式，首先调用CascadeClassifier函数，顾名思义这个函数跟分类有关
# 人脸识别就是机器学习领域中的分类任务
# 参数由两部分构成，中间用加号连接起来
# 第一部分是cv2.data.haarcascades，第二部分是一个xml文件名
# xml文件名上自第一个连字符开始也有提示信息frontalface_default，表示人脸识别功能引擎
# 后面还会讲到人眼识别和微笑识别，只需改变上述的提示信息即可调用识别引擎
face_engine = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 利用识别引擎来检测图片:调用识别引擎里的检测方法detectMultiScale
# 参数中主要有三个部分：检测的对象（即本地图片）、检测的精度（建议1.3到2.5之间的浮点数）以及检测的范围（值越小范围越小）
# 以上为函数里参数设置的基本描述，望大家自行调试
# 返回一个可迭代对象
faces = face_engine.detectMultiScale(img, 1.5, 4)
# 画出人脸的矩形区域
# 简单介绍一下可迭代对象的内容
# faces是引擎检测出来的每个面部区域的有效信息
# 包括坐标和宽度高度
# x，y为左上角的坐标，w,h为面部区域的宽度和高度
# 调用rectangle函数在图片上绘制矩形
# rectangle里的参数：指定的图片、左上角坐标、右下角坐标、BGR颜色设置矩形的边框颜色、边框宽度的设置
for (x, y, w, h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# 窗口显示图片:窗口名称、图片
cv2.imshow('Face_recognition',img)
# 安全退出窗口
cv2.waitKey(0)
