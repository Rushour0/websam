# Generates src/__fixtures__/add.onnx: a single-op ONNX graph computing
# c = a + b for float32 tensors of shape [1], opset 18.
#
# Run from packages/core:
#   uv run --with onnx python scripts/make-fixture.py
#
# The fixture is committed so tests never depend on Python at test time.
import os

import onnx
from onnx import TensorProto, helper


def main() -> None:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1])
    c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [1])
    node = helper.make_node("Add", ["a", "b"], ["c"], name="add0")
    graph = helper.make_graph([node], "websam_add_fixture", [a, b], [c])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="websam-make-fixture",
    )
    # Pin IR version for broad onnxruntime compatibility (opset 18 needs >= 8).
    model.ir_version = 10
    onnx.checker.check_model(model)

    out = os.path.join(os.path.dirname(__file__), "..", "src", "__fixtures__", "add.onnx")
    out = os.path.normpath(out)
    onnx.save(model, out)
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
