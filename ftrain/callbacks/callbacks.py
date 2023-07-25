import lightning.pytorch.callbacks as C  # noqa: D100, N812

MIN_DELTA = 0.001

__all__ = ["simple_callbacks", "MIN_DELTA"]


simple_callbacks = {
    "checkpoint": C.ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, verbose=True
    ),
    "early_stopping": C.EarlyStopping(
        monitor="val_loss", mode="min", patience=5, min_delta=MIN_DELTA
    ),
    "lr_monitor": C.LearningRateMonitor(logging_interval="epoch"),
    "module_summary": C.RichModelSummary(max_depth=-1),
    "progress_bar": C.RichProgressBar(),
}
