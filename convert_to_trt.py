import tensorrt as trt
import argparse
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx_path', type=str, required=True)
    p.add_argument('--engine_path', type=str, default='model_fp16.engine')
    p.add_argument('--fp16', action='store_true')
    return p.parse_args()


def build_engine(onnx_path, engine_path, fp16):
    logger = trt.Logger(trt.Logger.WARNING)
    print(f"📄 Leyendo ONNX: {onnx_path}")

    with trt.Builder(logger) as builder, \
         builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, logger) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)   # 1GB

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("⚡ FP16 ACTIVADO")

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("❌ Error al parsear el ONNX")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return

        # Perfil para batch dinámico
        profile = builder.create_optimization_profile()
        inp = network.get_input(0).name
        shape = (1, 3, 224, 224)
        profile.set_shape(inp, shape, shape, shape)
        config.add_optimization_profile(profile)

        print("🔧 Construyendo engine TensorRT...")
        engine = builder.build_engine(network, config)
        if engine is None:
            print("❌ Error al construir TensorRT")
            return

        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())

        print(f"✔ Engine guardado en: {engine_path}")


def main():
    args = parse_args()
    build_engine(args.onnx_path, args.engine_path, args.fp16)


if __name__ == "__main__":
    main()

