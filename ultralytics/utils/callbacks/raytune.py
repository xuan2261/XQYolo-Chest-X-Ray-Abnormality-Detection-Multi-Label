# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import SETTINGS

try:
    assert SETTINGS["raytune"] is True  # verify integration is enabled
    import ray
    from ray import tune
    from ray.air import session

except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    """Sends training metrics to Ray Tune at end of each epoch."""
    # if ray.train._internal.session.get_session():  # replacement for deprecated ray.tune.is_session_enabled()
    #     metrics = trainer.metrics
    #     metrics["epoch"] = trainer.epoch
    #     session.report(metrics)
    try:
        # Thử phương thức mới được gợi ý
        has_session = ray.train.get_session()
    except (AttributeError, ImportError):
        try:
            # Thử phương thức cũ (phòng hờ)
            # has_session = ray.tune.is_session_enabled() 
            has_session = ray.train._internal.session._get_session() 
        except (AttributeError, ImportError):
            has_session = False
    
    if has_session:
        metrics = trainer.metrics
        metrics["epoch"] = trainer.epoch
        session.report(metrics)


callbacks = (
    {
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    if tune
    else {}
)



# # Ultralytics YOLO 🚀, AGPL-3.0 license

# from ultralytics.utils import SETTINGS, LOGGER

# # --- Cải thiện try-except để xử lý lỗi import tốt hơn ---
# try:
#     # Chỉ kiểm tra setting nếu muốn bật tích hợp raytune
#     if SETTINGS.get("raytune", False): # Lấy giá trị setting an toàn hơn
#         import ray
#         from ray.air import session # Chỉ cần session để report
#         from ray import tune # Có thể cần tune cho is_session_enabled fallback
#         # Xác minh rằng ray, session, tune đã được import thành công
#         assert ray is not None
#         assert session is not None
#         # assert tune is not None # Không nhất thiết cần tune nếu get_session hoạt động
#         LOGGER.info("RayTune integration enabled.")
#         RAYTUNE_AVAILABLE = True
#     else:
#         # Nếu setting là False, không cần import ray
#         RAYTUNE_AVAILABLE = False
#         ray = None
#         session = None
#         tune = None

# except (ImportError, AssertionError, ModuleNotFoundError):
#     # Nếu có lỗi import hoặc setting bị thiếu/sai, tắt tích hợp
#     RAYTUNE_AVAILABLE = False
#     ray = None
#     session = None
#     tune = None
#     # Không cần log lỗi ở đây, chỉ cần đảm bảo không bị crash


# def on_fit_epoch_end(trainer):
#     """Sends training metrics to Ray Tune at end of each epoch if session is active."""

#     # Nếu RayTune không khả dụng (do lỗi import hoặc setting), không làm gì cả
#     if not RAYTUNE_AVAILABLE:
#         return

#     has_session = False # Mặc định là không có session
#     current_session = None
#     try:
#         # --- SỬA LỖI: Sử dụng API công khai ray.train.get_session() ---
#         current_session = ray.train.get_session()
#         has_session = current_session is not None
#         # --- KẾT THÚC SỬA LỖI ---

#     # Ngoại lệ này nên bắt các lỗi nếu ray không được cài đặt đúng cách
#     # hoặc API thay đổi hoàn toàn
#     except (AttributeError, ImportError, NameError) as e:
#          # Có thể thêm log ở đây nếu muốn debug kỹ hơn
#          # LOGGER.warning(f"Could not check Ray session using ray.train.get_session(): {e}")
#          # Thử fallback với ray.tune nếu có
#          if tune is not None:
#              try:
#                  has_session = ray.tune.is_session_enabled()
#              except (AttributeError, ImportError, NameError):
#                  has_session = False # Cả hai cách đều thất bại
#          else:
#             has_session = False # tune không được import, không thể fallback


#     # Nếu có session đang hoạt động, gửi metrics
#     if has_session and current_session is not None: # Đảm bảo current_session tồn tại
#         try:
#             metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
#             # Đảm bảo metrics không rỗng và epoch tồn tại
#             if hasattr(trainer, 'epoch') and metrics:
#                 metrics["epoch"] = trainer.epoch
#                 session.report(metrics) # Sử dụng session đã import từ ray.air
#             # else: # Log nếu không có metrics hoặc epoch
#             #     LOGGER.warning("RayTune callback: No metrics or epoch found in trainer to report.")
#         except Exception as e:
#             # Bắt lỗi chung khi gửi report phòng trường hợp API session.report thay đổi
#             LOGGER.warning(f"RayTune callback: Failed to report metrics: {e}")


# # Callback dictionary, chỉ thêm callback nếu RayTune khả dụng
# callbacks = (
#     {
#         "on_fit_epoch_end": on_fit_epoch_end,
#     }
#     if RAYTUNE_AVAILABLE # Chỉ thêm nếu import và check setting thành công
#     else {}
# )