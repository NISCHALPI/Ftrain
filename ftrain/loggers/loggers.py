from pytorch_lightning.loggers import TensorBoardLogger  # noqa: D100

__all__ = ["get_tensor_board_logger"]


def get_tensor_board_logger(  # noqa: ANN201, D103
    save_dir: str | None = None,
    name: str = "tfb_logs",
    *args,
    **kwargs,  # noqa: ANN002, ANN003
):
    return TensorBoardLogger(
        save_dir,
        name,
        0,
        True,
        *args,  # noqa: B026
        **kwargs,
    )
