import torch  # noqa: D100
from pytorch_lightning.profilers import PyTorchProfiler

__all__ = ["get_pytorch_profiler"]


def get_pytorch_profiler(  # noqa: ANN201, D103
    dirpath: str = None,
    filename: str = None,
    *args,
    **kwargs,  # noqa: ANN002, ANN003
):
    return PyTorchProfiler(
        dirpath=dirpath,
        filename=filename,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "dirpath/tfb_logs"
        ),
    )
