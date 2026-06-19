import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

def extract_resnet50_features(image_path):
    # ============ 图像预处理 ============
    # 定义与ImageNet预训练模型匹配的标准化参数[6,7](@ref)
    preprocess = transforms.Compose([
        transforms.Resize(256),               # 短边调整到256像素
        transforms.CenterCrop(224),           # 中心裁剪224x224
        transforms.ToTensor(),                # 转换为张量[0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],       # ImageNet标准化均值
            std=[0.229, 0.224, 0.225]         # ImageNet标准化方差
        )
    ])
    
    # 加载并预处理图像[6](@ref)
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image)           # 输出形状 [3,224,224]
    input_batch = input_tensor.unsqueeze(0)    # 添加批次维度 [1,3,224,224]

    # ============ 模型加载与修改 ============
    # 加载预训练ResNet-50[3,6](@ref)
    model = models.resnet50(pretrained=True)
    
    # 替换最后的全连接层为Identity[3](@ref)
    model.fc = torch.nn.Identity()            # 输出全局平均池化后的2048维特征
    
    # 设置为评估模式（禁用Dropout/BatchNorm更新）[6](@ref)
    model.eval()

    # ============ 特征提取 ============
    with torch.no_grad():                      # 禁用梯度计算节省内存[7](@ref)
        features = model(input_batch)         # 前向传播
        
    # 输出特征向量形状验证
    print(f"特征向量维度: {features.shape}")  # 应为 torch.Size([1, 2048])

    return features.squeeze(0)                # 移除批次维度返回2048维向量

# 使用示例
if __name__ == "__main__":
    image_path = "/mnt/HDD1/lc23/tyk/pgformer/dog/dog.0.jpg"             # 替换为实际图片路径
    feature_vector = extract_resnet50_features(image_path)
    
    # 特征应用示例
    print("示例特征值:", feature_vector[:5].tolist())  # 打印前5个特征值
    torch.save(feature_vector, "image_features.pt")    # 保存特征到文件