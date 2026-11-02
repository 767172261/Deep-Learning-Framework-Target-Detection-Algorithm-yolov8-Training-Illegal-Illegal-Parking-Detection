深度学习框架目标检测算法yolov8训练 非法违规停车检测数据集  建立深度学习yolov8违规停车检测系统
@[toc]
以下文字及代码仅供参考学习使用。


非法违规停车检测数据集 适用于yolo目标检测
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b177e2480f6f4191aa09111ee7f21afe.png)

训练测试验证 标签为txt格式
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a7cecf866da344368b547e1b1c922f7f.png)

类别有4类，分别为： ['available', 'full', 'illegal', 'legal']

训练集：1487张
测试集：216张
验证集：431张

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/df93443c5cc34655bb6b11ab008dd516.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0943ed616c2e43548bc52ed3cb51342c.png)
# 非法停车检测数据集 YOLO 目标检测训练

## 一、数据集结构
确保数据集按照以下结构组织：
```
parking_dataset/
├── images/
│   ├── train/
│   ├── test/
│   └── val/
└── labels/
    ├── train/
    ├── test/
    └── val/
```

## 二、创建 data.yaml 文件
```yaml
train: ./parking_dataset/images/train
val: ./parking_dataset/images/val
test: ./parking_dataset/images/test

nc: 4
names: ['available', 'full', 'illegal', 'legal']
```

## 三、下载YOLOv8预训练模型
```bash
# 下载YOLOv8n模型（轻量级）
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## 四、训练脚本
```python
from ultralytics import YOLO
import datetime

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 获取当前时间作为训练标识
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 开始训练
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name=f'parking_detection_{current_time}',
    project='runs/parking',
    pretrained=True,
    optimizer='AdamW',  # 使用AdamW优化器
    lr0=0.01,          # 初始学习率
    lrf=0.01,          # 最终学习率
    momentum=0.937,    # 动量参数
    weight_decay=0.0005,  # 权重衰减
    warmup_epochs=3,    # 预热期
    box=0.05,           # box损失权重
    cls=0.5,            # 分类损失权重
    dfl=1.5,            # DFL损失权重
    hsv_h=0.015,        # 色彩增强参数
    hsv_s=0.7, 
    hsv_v=0.4,
    degrees=0,          # 不使用旋转增强
    translate=0.1,      # 平移增强
    scale=0.5,          # 缩放增强
    shear=0,            # 剪切增强
    perspective=0.0001, # 透视变换
    flipud=0.5,         # 上下翻转
    fliplr=0.5,         # 左右翻转
    mosaic=1.0,         # 马赛克增强
    mixup=0.2,          # MixUp增强
    copy_paste=0.3,     # Copy-Paste增强
    close_mosaic=10,    # 关闭马赛克增强的epoch数
    verbose=True
)
```

## 五、验证脚本
```python
# 加载最佳模型进行验证
best_model = YOLO('runs/parking/parking_detection_20230815_143045/best.pt')

# 在验证集上评估模型性能
metrics = best_model.val()

# 打印主要指标
print(f"mAP@0.5-0.95: {metrics.box.map:.4f}")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"Recall@max1: {metrics.box.recall[0]:.4f}")
print(f"Precision@max1: {metrics.box.precision[0]:.4f}")

# 可视化一些验证结果
results = best_model('parking_dataset/images/val/', stream=True)
for i, r in enumerate(results):
    if i < 5:  # 显示前5个验证结果
        r.plot()
        plt.show()
```

## 六、推理脚本
```python
# 加载最佳模型
best_model = YOLO('runs/parking/parking_detection_20230815_143045/best.pt')

# 对单张图片进行推理
results = best_model('parking_dataset/images/test/example.jpg')

# 显示结果
results[0].plot()
plt.show()
```

## 七、非法停车检测系统开发建议

### 1. 系统设计思路：
- 使用YOLOv8进行车辆和停车位状态检测
- 结合规则引擎判断是否存在非法停车行为
- 添加跟踪算法识别长期停留车辆

### 2. 规则引擎实现示例：
```python
def check_illegal_parking(detections, parking_spots):
    """
    检测非法停车行为
    
    参数:
    detections: 检测结果，包含'available', 'full', 'illegal', 'legal'类别
    parking_spots: 停车位区域坐标
    
    返回:
    illegal_vehicles: 非法停车的车辆位置列表
    """
    illegal_vehicles = []
    
    # 将检测结果转换为更易处理的格式
    detection_dict = {
        'available': [],
        'full': [],
        'illegal': [],
        'legal': []
    }
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls_name = list(detection_dict.keys())[int(cls)]
        detection_dict[cls_name].append((x1, y1, x2, y2, conf))
    
    # 检查是否有车辆停在非法区域
    for x1, y1, x2, y2, conf in detection_dict['illegal']:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 如果车辆中心位于任何停车位区域内，则不是非法停车
        parked = False
        for spot in parking_spots:
            sx1, sy1, sx2, sy2 = spot
            if (sx1 < center_x < sx2) and (sy1 < center_y < sy2):
                parked = True
                break
        
        if not parked:
            illegal_vehicles.append((x1, y1, x2, y2, conf))
    
    return illegal_vehicles

# 使用示例
detections = results.xyxy[0].cpu().numpy()  # 获取第一个图像的检测结果
parking_spots = [(100, 100, 200, 200), (300, 100, 400, 200)]  # 示例停车位区域
illegal_vehicles = check_illegal_parking(detections, parking_spots)

print(f"检测到 {len(illegal_vehicles)} 辆非法停车车辆")
```


