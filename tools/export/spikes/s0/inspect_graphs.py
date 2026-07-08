"""S0 spike: enumerate IO contract of the community sam3-tracker ONNX graphs.

Run: uv run --with onnx python inspect_graphs.py
Loads graph protobufs WITHOUT external weight data (load_external_data=False).
"""
import onnx
from onnx import TensorProto

DT = {v: k for k, v in TensorProto.DataType.items()}


def dims(t):
    d = []
    for dim in t.type.tensor_type.shape.dim:
        if dim.HasField("dim_param") and dim.dim_param:
            d.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            d.append(dim.dim_value)
        else:
            d.append("?")
    return d


def show(path):
    m = onnx.load(path, load_external_data=False)
    print(f"\n=== {path} ===")
    print("ir_version:", m.ir_version)
    print("producer:", m.producer_name, m.producer_version)
    print("opsets:", {(o.domain or "ai.onnx"): o.version for o in m.opset_import})
    print("-- inputs --")
    for t in m.graph.input:
        print(f"  {t.name}: {DT[t.type.tensor_type.elem_type]} {dims(t)}")
    print("-- outputs --")
    for t in m.graph.output:
        print(f"  {t.name}: {DT[t.type.tensor_type.elem_type]} {dims(t)}")


show("vision_encoder.onnx")
show("prompt_encoder_mask_decoder.onnx")
