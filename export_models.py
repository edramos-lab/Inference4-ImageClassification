import argparse
import os
import torch
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import onnxruntime as ort

# ============================================================
#     CRUCIAL FIX — FORCE LEGACY ONNX EXPORTER ON JETSON
# ============================================================
torch.onnx.enable_export()       # Force legacy exporter
torch.onnx.dynamo_export = False # Disable dynamo → disables onnxscript
print("⚙ Using LEGACY ONNX EXPORTER (Jetson-safe)", flush=True)
# ============================================================


"""
Flexible Model Export Script for ONNX and TensorRT
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Export trained models to ONNX and TensorRT formats.')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='efficientnet_b0')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type:int, default=1)
    parser.add_argument('--output_dir', type=str, default='exported_models')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--class_names', type=str, nargs='+',
                        default=['class_0', 'class_1', 'class_2', 'class_3'])
    return parser.parse_args()


def load_model(model_path, model_name, num_classes, device):
    print(f"Creating model: {model_name}")
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


# ============================================================
#       FIXED ONNX EXPORT — LEGACY EXPORTER ONLY
# ============================================================
def export_to_onnx(model, output_path, input_size, batch_size, device):
    print(f"🚀 Exporting ONNX using LEGACY torch.onnx._export(): {output_path}")

    model.eval()

    dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)

    # --- Use LEGACY EXPORT FUNCTION ---
    torch.onnx._export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print("✔ Legacy ONNX export completed (torch.onnx._export).")



# ============================================================
#            YOUR ORIGINAL TENSORRT EXPORTER
# ============================================================
def export_to_tensorrt(onnx_path, output_path, fp16=False, int8=False):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ Using FP16 mode")

        if int8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            print("⚡ INT8 enabled")

        print(f"Parsing ONNX: {onnx_path}")
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("❌ Failed to parse ONNX")

        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        shape = (1, 3, 224, 224)
        profile.set_shape(input_name, min=shape, opt=shape, max=(4, 3, 224, 224))
        config.add_optimization_profile(profile)

        print("🔧 Building TensorRT engine...")
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("❌ TensorRT engine build failed")

        with open(output_path, 'wb') as f:
            f.write(engine)

        print("✅ TensorRT engine exported successfully!")


# ============================================================
#         YOUR ONNX & TRT TEST FUNCTIONS (UNCHANGED)
# ============================================================
def test_onnx_inference(onnx_path, test_image_path, class_names):
    print("Testing ONNX inference...")
    ort_sess = ort.InferenceSession(onnx_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(test_image_path).convert('RGB')
    inp = transform(image).unsqueeze(0).numpy()
    output = ort_sess.run(None, {'input': inp})[0]
    pred = np.argmax(output)
    print(f"ONNX Prediction: {class_names[pred]}")
    return pred


def test_tensorrt_inference(engine_path, test_image_path, class_names):
    print("Testing TensorRT inference...")
    logger = trt.Logger(trt.Logger.WARNING)

    with open(engine_path, 'rb') as f:
        engine_data = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    ctx = engine.create_execution_context()

    input_shape = (1, 3, 224, 224)
    output_shape = (1, len(class_names))

    d_input = cuda.mem_alloc(np.prod(input_shape) * 4)
    d_output = cuda.mem_alloc(np.prod(output_shape) * 4)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(test_image_path).convert('RGB')
    inp = transform(image).unsqueeze(0).numpy().astype(np.float32)

    cuda.memcpy_htod(d_input, inp)
    ctx.execute_v2([int(d_input), int(d_output)])

    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)
    pred = np.argmax(output)
    print(f"TensorRT Prediction: {class_names[pred]}")
    return pred


# ============================================================
#                        MAIN
# ============================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    class_names = args.class_names
    print(f"📊 Classes: {class_names}")

    if len(class_names) != args.num_classes:
        print("⚠️ num_classes mismatch → fixing automatically")
        args.num_classes = len(class_names)

    model = load_model(args.model_path, args.model_name, args.num_classes, device)

    base_name = os.path.splitext(os.path.basename(args.model_path))[0]
    onnx_path = os.path.join(args.output_dir, f"{base_name}.onnx")
    trt_path = os.path.join(args.output_dir, f"{base_name}.engine")

    export_to_onnx(model, onnx_path, args.input_size, args.batch_size, device)
    export_to_tensorrt(onnx_path, trt_path, args.fp16, args.int8)

    print("\n🎉 Export complete!")
    print(f"ONNX → {onnx_path}")
    print(f"TRT  → {trt_path}")


if __name__ == "__main__":
    main()

