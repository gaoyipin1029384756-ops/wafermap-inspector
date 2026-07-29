"""
Streamlit 网页应用
晶圆缺陷检测系统 - 上传图片，AI 自动识别缺陷类型
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys

# 添加 src 到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import get_model
from data_loader import SyntheticWaferDataset, generate_wafer_map


# 页面配置
st.set_page_config(
    page_title="WaferMap Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
        .defect-card {
        background-color: #1e3a5f;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: #ffffff;
    }
    .defect-card h3 {
        color: #ffffff;
        margin: 0 0 10px 0;
    }
    .defect-card p {
        color: #e0e0e0;
        margin: 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """加载训练好的模型"""
    model_path = os.path.join('models', 'best_model.pth')
    
    if not os.path.exists(model_path):
        st.error("❌ 找不到模型文件！请先运行训练脚本: `python src/train.py`")
        return None, None
    
    checkpoint = torch.load(model_path, map_location='cpu')
    class_names = checkpoint['class_names']
    
    model = get_model(num_classes=len(class_names), device='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, class_names


def preprocess_image(img):
    """预处理图像"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return transform(img).unsqueeze(0)


def predict(model, image_tensor, class_names):
    """预测缺陷类型"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probabilities, 1)
    
    predicted_class = class_names[pred.item()]
    confidence = conf.item() * 100
    all_probs = probabilities.squeeze().numpy()
    
    return predicted_class, confidence, all_probs


def generate_sample(defect_type):
    """生成示例晶圆图"""
    wafer_map = generate_wafer_map(size=64, defect_type=defect_type)
    
    rgb_img = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb_img[wafer_map == 0] = [0, 0, 0]
    rgb_img[wafer_map == 1] = [0, 200, 0]
    rgb_img[wafer_map == 2] = [255, 0, 0]
    
    return Image.fromarray(rgb_img)


def main():
    # 标题
    st.markdown('<div class="main-header">🔬 WaferMap Inspector</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">基于深度学习的晶圆缺陷智能检测系统</div>', 
                unsafe_allow_html=True)
    
    # 加载模型
    model, class_names = load_model()
    if model is None:
        st.info("💡 提示：第一次使用需要先训练模型。在终端运行: `python src/train.py`")
        return
    
    # ========== 侧边栏 ==========
    with st.sidebar:
        st.header("⚙️ 控制面板")
        
        st.subheader("1. 生成模拟晶圆")
        selected_defect = st.selectbox(
            "选择缺陷类型",
            class_names,
            key="defect_selector"
        )
        
        if st.button("🎲 生成示例", use_container_width=True):
            st.session_state['sample'] = generate_sample(selected_defect)
            st.session_state['sample_type'] = selected_defect
        
        st.divider()
        
        st.subheader("2. 上传图片")
        uploaded_file = st.file_uploader(
            "上传晶圆图 (JPG/PNG)",
            type=['jpg', 'jpeg', 'png'],
            key="file_uploader"
        )
        
        st.divider()
        
        st.subheader("📊 缺陷类型说明")
        defect_info = {
            "Center": "缺陷集中在晶圆中心区域",
            "Donut": "缺陷呈环形分布（甜甜圈状）",
            "Edge-Loc": "缺陷位于晶圆边缘局部位置",
            "Edge-Ring": "缺陷沿晶圆边缘呈环形分布",
            "Loc": "缺陷集中在某一局部区域",
            "Random": "缺陷随机分散分布",
            "Scratch": "缺陷呈线条状（划痕）",
            "Near-full": "大面积缺陷，几乎覆盖整个晶圆",
            "none": "无明显缺陷模式"
        }
        
        for name, desc in defect_info.items():
            with st.expander(f"🔹 {name}"):
                st.write(desc)
    
    # ========== 主界面 ==========
    # 确定要检测的图像（在 sidebar 外处理，避免变量丢失）
    image_to_predict = None
    image_caption = ""
    
    if 'sample' in st.session_state:
        image_to_predict = st.session_state['sample']
        image_caption = f"模拟晶圆 - {st.session_state.get('sample_type', '')}"
    
    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
        image_caption = "上传的晶圆图"
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📷 输入图像")
        if image_to_predict is not None:
            st.image(image_to_predict, caption=image_caption, use_column_width=True)
        else:
            st.info("👈 请在左侧生成示例或上传图片")
    
    with col2:
        st.header("🎯 检测结果")
        
        if image_to_predict is not None:
            img_tensor = preprocess_image(image_to_predict)
            pred_class, confidence, all_probs = predict(model, img_tensor, class_names)
            
            conf_class = "confidence-high" if confidence > 80 else "confidence-medium" if confidence > 50 else "confidence-low"
            st.markdown(f"""
            <div class="defect-card">
                <h3>预测结果: {pred_class}</h3>
                <p>置信度: <span class="{conf_class}">{confidence:.2f}%</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("各类别概率分布")
            sorted_indices = np.argsort(all_probs)[::-1]
            for idx in sorted_indices[:5]:
                prob = all_probs[idx] * 100
                name = class_names[idx]
                st.progress(int(prob), text=f"{name}: {prob:.2f}%")
            
            st.divider()
            st.subheader("💡 结果解读")
            
            if pred_class == "none":
                st.success("✅ 未检测到明显缺陷模式，晶圆质量良好。")
            elif pred_class == "Center":
                st.warning("⚠️ 中心区域缺陷：可能由光刻机中心曝光不均或刻蚀液分布问题导致。")
            elif pred_class == "Edge-Ring":
                st.warning("⚠️ 边缘环形缺陷：常见于薄膜沉积不均匀或边缘刻蚀过度。")
            elif pred_class == "Scratch":
                st.error("🚨 划痕缺陷：可能由晶圆搬运过程中的机械损伤或设备摩擦导致，需立即检查工艺流程。")
            else:
                st.info(f"ℹ️ {pred_class} 类型缺陷：建议结合具体工艺流程进一步分析根因。")
        else:
            st.info("👈 请在左侧生成示例或上传图片开始检测")
    
    # 页脚
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>WaferMap Inspector | 基于 PyTorch + Streamlit 构建</p>
        <p>微电子 + AI 交叉项目 | 适合专升本作品集展示</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()