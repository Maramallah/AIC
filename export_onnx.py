import os
import torch
import argparse
import sys

# استدعاء دوال التهيئة من الكود الذي استخدمناه مسبقاً
from live_demo import setup_lightfc_repo, load_config, load_model_v6_exact

def parse_args():
    parser = argparse.ArgumentParser(description="Export LightFC to ONNX")
    parser.add_argument("--repo-dir", type=str, default="third_party/LightFC")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_checkpoint.pth.tar")
    parser.add_argument("--save-dir", type=str, default="checkpoints/onnx", help="Directory to save ONNX files")
    return parser.parse_args()

def export_to_onnx():
    args = parse_args()
    
    # 1. إعداد البيئة والموديل (باستخدام نفس الإعدادات الخاصة بك)
    repo_dir = setup_lightfc_repo(args.repo_dir)
    cfg = load_config(repo_dir)
    model = load_model_v6_exact(cfg, args.checkpoint)
    
    # وضع الموديل في حالة التقييم
    model.eval().cpu() # تصدير ONNX يفضل أن يتم على الـ CPU لتجنب أخطاء بعض العمليات
    
    os.makedirs(args.save_dir, exist_ok=True)
    backbone_path = os.path.join(args.save_dir, "lightfc_backbone.onnx")
    network_path = os.path.join(args.save_dir, "lightfc_network.onnx")

    print("🚀 Starting ONNX Export for MTC-AIC4 UAV Tracking...")

    # 2. تجهيز مدخلات وهمية (Dummy Inputs) بنفس أبعاد الصور التي يستخدمها الموديل
    # الافتراضي في كودك: Template=128x128, Search=256x256
    dummy_template = torch.randn(1, 3, 128, 128)
    dummy_search = torch.randn(1, 3, 256, 256)

    # 3. تصدير الـ Backbone
    print("⏳ Exporting Backbone...")
    try:
        # استخراج ميزات الـ Template
        z_features = model.forward_backbone(dummy_template)
        
        # بعض الموديلات تخرج Tuple أو Dict، نتعامل معها هنا كـ Tensor للتصدير
        torch.onnx.export(
            model.backbone,               # الجزء الخاص بالـ backbone فقط
            dummy_template,               # المدخلات الوهمية
            backbone_path,                # مسار الحفظ
            export_params=True,
            opset_version=12,             # إصدار متوافق مع TensorRT
            do_constant_folding=True,
            input_names=['template'],
            output_names=['template_features'],
        )
        print(f"✅ Backbone successfully exported to: {backbone_path}")
    except Exception as e:
        print(f"❌ Error exporting Backbone: {e}")

    # 4. تصدير الـ Tracking Network (Head)
    print("⏳ Exporting Tracking Network...")
    try:
        # نحتاج إلى تمرير ميزات الـ Template (z) وميزات الـ Search (x)
        # إذا كان الكود لديك يستخدم forward_tracking بشكل مباشر:
        class TrackingHeadWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, z, x):
                return self.model.forward_tracking(z, x)
                
        wrapper = TrackingHeadWrapper(model)
        
        torch.onnx.export(
            wrapper, 
            (z_features, dummy_search),   # تمرير الميزات السابقة مع الفريم الجديد
            network_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['template_features', 'search_region'],
            output_names=['score_map', 'size_map', 'offset_map']
        )
        print(f"✅ Tracking Network successfully exported to: {network_path}")
    except Exception as e:
        print(f"❌ Error exporting Tracking Network: {e}")

    print("🎉 Export complete! You are ready for TensorRT conversion on the Jetson.")

if __name__ == "__main__":
    export_to_onnx()