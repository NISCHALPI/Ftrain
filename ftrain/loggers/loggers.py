from lightning.pytorch.loggers import TensorBoardLogger  # noqa: D100

__all__ = ["get_tensor_board_logger"]


def get_tensor_board_logger(  # noqa: ANN201, D103
    save_dir: str = None,
    name: str = "tfb_logs",
    *args,
    **kwargs,  # noqa: ANN002, ANN003
):
    return TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=0,
        log_graph=True,
        *args,  # noqa: B026
        **kwargs,
    )
