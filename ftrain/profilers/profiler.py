import torch  # noqa: D100
from pytorch_lightning.profilers import PyTorchProfiler

__all__ = ["get_pytorch_profiler"]


def get_pytorch_profiler(
    dirpath: str | None = None,
    filename: str | None = None,
) -> PyTorchProfiler:
    """Get a PyTorch profiler.

    Args:
        dirpath (str, optional): Path to saving directory. Defaults to None.
        filename (str, optional): Name of the file. Defaults to None.

    Returns:
        _type_: _description_
    """
    return PyTorchProfiler(
        dirpath=dirpath,
        filename=filename,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("dirpath/tfb_logs"),
    )
