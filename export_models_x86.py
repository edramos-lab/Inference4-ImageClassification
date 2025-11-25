import torch
import timm
import argparse
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--model_name', type=str, default='efficientvit_b0')
    p.add_argument('--num_classes', type=int, default=4)
    p.add_argument('--input_size', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--output_path', type=str, default='export.onnx')
    return p.parse_args()


def export_onnx(model, dummy_input, output_path):
    print("🚀 Exportando a ONNX usando exportador LEGACY (CPU/x86)...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,          # TRT del Jetson soporta 12
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    print(f"✔ Exportación completada: {output_path}")


def main():
    args = parse_args()

    device = "cpu"
    print(f"📦 Cargando modelo {args.model_name}")

    model = timm.create_model(
        args.model_name,
        pretrained=False,
        num_classes=args.num_classes
    )

    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(args.batch_size, 3, args.input_size, args.input_size)

    out_dir = os.path.dirname(args.output_path)
    if out_dir != "" and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    export_onnx(model, dummy, args.output_path)


if __name__ == "__main__":
    main()

