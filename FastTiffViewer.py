import sys
import os
import logging
from logging.handlers import RotatingFileHandler
import time
import gc
import struct
import shutil
from pathlib import Path
from collections import OrderedDict

from PySide6.QtCore import (
    Qt,
    QObject,
    Signal,
    Slot,
    QRunnable,
    QThreadPool,
    QByteArray,
    QMimeData,
    QPointF,
    QRect,
    QRectF,
    QSize,
    QStandardPaths,
    QTimer,
    QUrl,
)
from PySide6.QtGui import QAction, QBrush, QColor, QCursor, QGuiApplication, QIcon, QImage, QPainter, QPen
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QMainWindow,
    QMenu,
    QMessageBox,
    QLabel,
    QLineEdit,
    QProgressBar,
    QStyle,
    QSystemTrayIcon,
)
import pyvips
from PIL import Image as PILImage


def _default_log_file_path() -> Path:
    # 配布版EXEの配置先に依存せず、ユーザーが書き込める固定フォルダへ保存する
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "FastTiffViewer" / "fasttiffviewer_debug.log"
    return Path(__file__).with_name("fasttiffviewer_debug.log")


LOG_FILE_PATH = _default_log_file_path()
LOGGER = logging.getLogger("fasttiffviewer")
ENABLE_DEBUG_LOGGING = os.getenv("TIFFVIEWER_DEBUG_LOG", "0").strip().lower() in {"1", "true", "yes", "on"}
LOG_LEVEL = logging.DEBUG if os.getenv("TIFFVIEWER_LOG_LEVEL", "INFO").strip().upper() == "DEBUG" else logging.INFO
LOG_FILE_MAX_BYTES = 5 * 1024 * 1024    # ログファイル1世代の最大サイズ
LOG_FILE_BACKUP_COUNT = 2               # 保持するバックアップログの世代数
# 通常はログを停止し、必要な場合だけTIFFVIEWER_DEBUG_LOG=1で有効化する

WINDOW_TITLE = "Fast TIFF Viewer v1.7.0"
INSTANCE_SERVER_NAME = "FastTiffViewer.Singleton.Main"

# 表示/デコード挙動の調整パラメータ
DEFAULT_WINDOW_SIZE = (1060, 800)       # アプリ起動時の初期ウィンドウサイズ(px)
MIN_WINDOW_SIZE = (495, 400)            # ウィンドウの最小許容サイズ(px)
CTRL_WHEEL_WINDOW_SCALE_BASE = 1.12     # Ctrl+ホイール1段あたりのウィンドウ拡縮倍率
FULLRES_IDLE_DELAY_MS = 280             # 通常操作後にfullresデコード要求を出すまでの待機時間(ms)
FULLRES_PAGE_CHANGE_DELAY_MS = 180      # ページ切替直後にfullres要求するまでの遅延(ms)
FULLRES_AFTER_PRELOAD_DELAY_MS = 80     # 全ページプレビュー先読み完了後にfullres要求する遅延(ms)
ZOOM_INTERACTION_IDLE_MS = 220          # ズーム連続操作を終了とみなす無操作時間(ms)
FULLRES_AFTER_ZOOM_IDLE_DELAY_MS = 40   # ズーム終了後にfullres要求するまでの遅延(ms)
FULLRES_AFTER_RESIZE_DELAY_MS = 140     # ウィンドウリサイズ後にfullres要求するまでの遅延(ms)
IDLE_PARTIAL_FULLRES_ENABLED = True     # アイドル時の段階的fullres（部分高解像度）要求を有効化するか
IDLE_PARTIAL_FULLRES_SCALE = 1.8        # 現在表示サイズに対して先行要求するfullres倍率
FIT_SNAP_TOLERANCE_RATIO = 1.01         # 縮小時にFitへ吸着する許容比率（1.0に近いほど厳密）
PREVIEW_RESIZE_UPDATE_DELAY_MS = 120    # リサイズ後にプレビュー再生成を実行するまでの遅延(ms)
PREVIEW_RESIZE_MIN_DELTA_PX = 64        # プレビュー再生成を行う最小サイズ差分(px)
PREVIEW_RESIZE_SMOOTH = True            # プレビュー再生成時に滑らか補間を使うか
PREVIEW_RESIZE_TARGET_SCALE = 1.0       # プレビュー再生成時の目標倍率（ビューポート基準）
WORKER_DRAIN_TIMEOUT_MS = 3000          # ファイル切替/解放時にバックグラウンド処理完了を待つ上限(ms)
SPREAD_SYNC_INTERVAL_MS = 16            # 見開き同期中のパン・ズーム通知間隔(ms)
SPREAD_SYNC_ZOOM_RATIO_EPSILON = 1e-4   # 同期更新をズームと判定する倍率差の相対許容値
CROP_BORDER_WIDTH_PX = 2.0              # トリミング枠の線幅(px)
CROP_HANDLE_SIZE_PX = 9.0               # トリミング枠ハンドルの表示サイズ(px)
CROP_EDGE_GRAB_WIDTH_PX = 10.0          # トリミング枠の辺をつかめる幅(px)
CROP_DRAG_START_DISTANCE_PX = 3         # クリック保存とDnD操作を分ける移動距離(px)
CROP_MIN_SIZE_PX = 2.0                  # トリミング保存を許可する最小サイズ(表示画像px)
CROP_SAVE_TEXT_MIN_WIDTH = 340          # 保存先テキストボックスの最小幅(px)
CROP_FULL_VIEWPORT_UPDATE = True        # トリミング枠表示中は全体再描画にしてパン時のちらつきを抑える
APP_CLOSE_SHORTCUT_KEY = Qt.Key_C       # ウィンドウを閉じるショートカットキー
DIFF_SHORTCUT_KEY = Qt.Key_D            # 差分検出を開始するショートカットキー
DIFF_TIFF_COMPRESSION = "lzw"           # 差分TIFFの圧縮形式
DIFF_FILE_REPLACE_MAX_ATTEMPTS = 8       # 共有違反時のファイル置換試行回数
DIFF_FILE_RETRY_BASE_DELAY_MS = 50       # ファイル置換再試行の初期待機時間(ms)
DIFF_FILE_RETRY_MAX_DELAY_MS = 500       # ファイル置換再試行の最大待機時間(ms)
DIFF_PROGRESS_BAR_WIDTH_PX = 220         # ステータスバー上の差分進捗バー幅(px)

TIFF_TAG_COMPRESSION = 259              # TIFF圧縮形式タグ
TIFF_TAG_BITS_PER_SAMPLE = 258          # TIFF 1サンプルあたりのビット数タグ
TIFF_COMPRESSION_TO_VIPS = {
    1: "none",
    4: "ccittfax4",
    5: "lzw",
    7: "jpeg",
    8: "deflate",
    32946: "deflate",
    32773: "packbits",
    34712: "jp2k",
    50000: "zstd",
    50001: "webp",
}


def _normalize_input_path(path_text: str) -> str:
    path = str(path_text).strip().strip('"')
    if not path:
        return ""

    # PyInstaller 経由の関連付け起動で file:// 形式が来るケースにも対応
    if path.lower().startswith("file://"):
        url = QUrl(path)
        if url.isValid() and url.isLocalFile():
            path = url.toLocalFile()
    return path


def _is_plain_key_event(event, key) -> bool:
    # Ctrl+Cなどの編集ショートカットと区別するため、修飾キーなしだけを対象にする
    return event.key() == key and event.modifiers() == Qt.NoModifier


def _build_ipc_message(args) -> str:
    if not args:
        return "PING"
    path = _normalize_input_path(args[0])
    if path:
        return f"OPEN\t{path}"
    return "PING"


def _send_ipc_message(message: str, timeout_ms: int = 500) -> bool:
    sock = QLocalSocket()
    sock.connectToServer(INSTANCE_SERVER_NAME)
    if not sock.waitForConnected(timeout_ms):
        return False

    payload = (message + "\n").encode("utf-8", errors="ignore")
    sock.write(payload)
    if not sock.waitForBytesWritten(timeout_ms):
        sock.abort()
        return False

    sock.flush()
    if sock.state() != QLocalSocket.UnconnectedState:
        sock.disconnectFromServer()
        if sock.state() != QLocalSocket.UnconnectedState:
            sock.waitForDisconnected(timeout_ms)
    return True


class SingleInstanceServer(QObject):
    def __init__(self, message_handler, parent=None):
        super().__init__(parent)
        self._message_handler = message_handler
        self._server = QLocalServer(self)
        self._server.newConnection.connect(self._on_new_connection)

    def start(self) -> bool:
        QLocalServer.removeServer(INSTANCE_SERVER_NAME)
        ok = self._server.listen(INSTANCE_SERVER_NAME)
        if ok:
            log_info("SingleInstanceServer listening name=%s", INSTANCE_SERVER_NAME)
        else:
            log_info("SingleInstanceServer listen failed name=%s err=%s", INSTANCE_SERVER_NAME, self._server.errorString())
        return ok

    def _on_new_connection(self):
        while self._server.hasPendingConnections():
            sock = self._server.nextPendingConnection()
            if sock is None:
                continue
            sock.readyRead.connect(lambda s=sock: self._on_socket_ready_read(s))
            sock.disconnected.connect(sock.deleteLater)

    def _on_socket_ready_read(self, sock):
        raw = bytes(sock.readAll()).decode("utf-8", errors="ignore")
        for line in raw.splitlines():
            message = line.strip()
            if not message:
                continue
            self._dispatch(message)

    def _dispatch(self, message: str):
        log_info("SingleInstanceServer received message=%s", message)
        if self._message_handler is not None:
            self._message_handler(message)


def setup_debug_logging():
    if not ENABLE_DEBUG_LOGGING:
        return
    if LOGGER.handlers:
        return
    try:
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            LOG_FILE_PATH,
            mode="a",
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        ))
        handler.setLevel(LOG_LEVEL)
        LOGGER.addHandler(handler)
        LOGGER.setLevel(LOG_LEVEL)
        LOGGER.propagate = False
        LOGGER.info(
            "===== session start pid=%s entry=%s argv=%s =====",
            os.getpid(),
            str(Path(sys.argv[0]).name),
            sys.argv[1:],
        )
    except Exception as e:
        print(f"[fasttiffviewer] failed to setup log file: {e}")


def log_debug(message, *args):
    if LOGGER.handlers:
        LOGGER.debug(message, *args)


def log_info(message, *args):
    if LOGGER.handlers:
        LOGGER.info(message, *args)


def log_exception(message, *args):
    if LOGGER.handlers:
        LOGGER.exception(message, *args)


def _size_text(size: QSize) -> str:
    if size is None or not size.isValid():
        return "invalid"
    return f"{size.width()}x{size.height()}"


def _img_text(img: QImage) -> str:
    if img is None or img.isNull():
        return "null"
    return f"{img.width()}x{img.height()}"


def _rect_text(rect: QRectF) -> str:
    if rect.isNull():
        return "null"
    return f"{rect.width():.1f}x{rect.height():.1f}"


def _find_app_icon_path() -> str:
    icon_names = ["FastTiffViewer.ico", "TiffViewer.ico"]
    candidates = []

    # PyInstaller（onefile / onedir）
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        base = Path(meipass) / "ico"
        for name in icon_names:
            candidates.append(base / name)

    # 実行ファイルの隣（PyInstaller onedir）
    base = Path(sys.executable).resolve().parent / "ico"
    for name in icon_names:
        candidates.append(base / name)

    # 通常のpython実行（スクリプト基準）
    base = Path(__file__).resolve().parent / "ico"
    for name in icon_names:
        candidates.append(base / name)

    for p in candidates:
        if p.is_file():
            return str(p)
    return ""


def _default_open_directory() -> str:
    candidates = []
    if os.name == "nt":
        one_drive = os.environ.get("OneDrive")
        user_profile = os.environ.get("USERPROFILE")
        if one_drive:
            candidates.append(Path(one_drive) / "Desktop")
        if user_profile:
            candidates.append(Path(user_profile) / "Desktop")

    candidates.append(Path.home() / "Desktop")
    candidates.append(Path.home())

    for p in candidates:
        if p.exists() and p.is_dir():
            return str(p)
    return str(Path.home())


def _default_desktop_directory() -> Path:
    candidates = []

    qt_desktop = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
    if qt_desktop:
        candidates.append(Path(qt_desktop))

    if os.name == "nt":
        one_drive = os.environ.get("OneDrive")
        user_profile = os.environ.get("USERPROFILE")
        if one_drive:
            candidates.append(Path(one_drive) / "Desktop")
        if user_profile:
            candidates.append(Path(user_profile) / "Desktop")

    candidates.append(Path.home() / "Desktop")
    candidates.append(Path.home())

    for p in candidates:
        try:
            if p.exists() and p.is_dir():
                return p
        except OSError:
            continue
    return Path.home()


def _desktop_copy_target_path(source_file_path: str) -> Path:
    return _default_desktop_directory() / Path(source_file_path).name


def _copy_file_to_desktop(source_file_path: str) -> Path:
    source_path = Path(source_file_path)
    if not source_path.is_file():
        raise FileNotFoundError(str(source_path))

    target_path = _desktop_copy_target_path(source_file_path)
    if target_path.exists():
        try:
            if source_path.samefile(target_path):
                return target_path
        except OSError:
            pass

        if target_path.is_dir():
            raise IsADirectoryError(str(target_path))

    # 現在表示中の元ファイルをコピーするだけで、ビューの読み込み先は変更しない
    shutil.copy2(source_path, target_path)
    return target_path


def _default_pictures_directory() -> Path:
    candidates = []

    qt_pictures = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
    if qt_pictures:
        candidates.append(Path(qt_pictures))

    if os.name == "nt":
        one_drive = os.environ.get("OneDrive")
        user_profile = os.environ.get("USERPROFILE")
        if one_drive:
            candidates.append(Path(one_drive) / "Pictures")
        if user_profile:
            candidates.append(Path(user_profile) / "Pictures")

    candidates.append(Path.home() / "Pictures")
    candidates.append(Path.home())

    for p in candidates:
        try:
            if p.exists() and p.is_dir():
                return p
        except OSError:
            continue
    return Path.home()


def _default_crop_save_text() -> str:
    # デフォルトはピクチャフォルダをディレクトリ指定として扱えるよう末尾区切りを付ける
    return str(_default_pictures_directory()) + os.sep


def _crop_timestamp_text() -> str:
    seconds = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1.0) * 1000.0)
    return f"{seconds}_{millis:03d}"


def _timestamped_crop_name(original_file_path: str, timestamp: str) -> str:
    original = Path(original_file_path)
    suffix = original.suffix or ".tif"
    return f"{original.stem}_{timestamp}{suffix}"


def _crop_text_has_trailing_separator(text: str) -> bool:
    cleaned = str(text or "").strip().strip('"')
    return bool(cleaned) and cleaned.endswith(("\\", "/"))


def _with_original_image_suffix(path: Path, original_suffix: str) -> Path:
    suffix = original_suffix or ".tif"
    current = path.suffix.lower()

    if not current:
        return path.with_suffix(suffix)

    # TIFF内の拡張子差だけは同一形式として許可する
    if suffix.lower() in {".tif", ".tiff"} and current in {".tif", ".tiff"}:
        return path

    if current != suffix.lower():
        return path.with_suffix(suffix)
    return path


def _resolve_crop_save_path(original_file_path: str, save_text: str, timestamp: str = "") -> Path:
    original = Path(original_file_path)
    original_suffix = original.suffix or ".tif"
    ts = timestamp or _crop_timestamp_text()
    raw = str(save_text or "").strip().strip('"')

    if not raw:
        return original.parent / _timestamped_crop_name(original_file_path, ts)

    expanded = os.path.expandvars(os.path.expanduser(raw))
    target = Path(expanded)
    is_directory_target = _crop_text_has_trailing_separator(raw)
    if not is_directory_target:
        try:
            is_directory_target = target.exists() and target.is_dir()
        except OSError:
            is_directory_target = False

    if is_directory_target:
        return target / _timestamped_crop_name(original_file_path, ts)

    # パスが省略されたファイル名だけの指定は、元画像と同じフォルダへ保存する
    if target.parent == Path("."):
        target = original.parent / target.name

    return _with_original_image_suffix(target, original_suffix)


def _diff_default_file_name(new_file_path: str) -> str:
    new_path = Path(new_file_path)
    suffix = new_path.suffix if new_path.suffix.lower() in {".tif", ".tiff"} else ".tif"
    return f"{new_path.stem}_dif{suffix}"


def _with_tiff_suffix(path: Path) -> Path:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return path
    return path.with_suffix(".tif")


def _resolve_diff_save_path(new_file_path: str, save_text: str) -> Path:
    """Crop save as欄を差分TIFFの保存先として解釈する。"""
    new_path = Path(new_file_path)
    default_name = _diff_default_file_name(new_file_path)
    raw = str(save_text or "").strip().strip('"')

    if not raw:
        return new_path.parent / default_name

    expanded = os.path.expandvars(os.path.expanduser(raw))
    target = Path(expanded)
    is_directory_target = _crop_text_has_trailing_separator(raw)
    if not is_directory_target:
        try:
            is_directory_target = target.exists() and target.is_dir()
        except OSError:
            is_directory_target = False

    if is_directory_target:
        return target / default_name

    # ファイル名だけの指定は、新画像と同じフォルダへ保存する
    if target.parent == Path("."):
        target = new_path.parent / target.name
    return _with_tiff_suffix(target)


def _normalized_absolute_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _vips_tiff_page_count(file_path: str) -> int:
    probe = pyvips.Image.new_from_file(
        file_path,
        access="sequential",
        sequential=True,
        autorotate=True,
        page=0,
        n=1,
    )
    if probe.get_typeof("n-pages") != 0:
        try:
            return max(1, int(probe.get("n-pages")))
        except (TypeError, ValueError):
            pass
    return 1


def _normalize_diff_vips_image(image: pyvips.Image) -> pyvips.Image:
    """差分演算用に画像を8bit RGBへ正規化する。"""
    normalized = image
    try:
        if str(normalized.interpretation) != "srgb":
            normalized = normalized.colourspace("srgb")
    except pyvips.Error:
        # 色空間情報がないTIFFもあるため、バンド数による正規化へフォールバックする
        pass

    if normalized.hasalpha():
        # 透明部分は白紙として比較する
        normalized = normalized.flatten(background=[255, 255, 255])

    if normalized.bands == 1:
        normalized = normalized.bandjoin(normalized).bandjoin(normalized)
    elif normalized.bands == 2:
        gray = normalized.extract_band(0)
        normalized = gray.bandjoin(gray).bandjoin(gray)
    elif normalized.bands > 3:
        normalized = normalized.extract_band(0, n=3)

    if normalized.format != "uchar":
        normalized = normalized.cast("uchar")
    return normalized.copy(interpretation="srgb")


def _white_diff_vips_image(width: int, height: int) -> pyvips.Image:
    base = pyvips.Image.black(width, height, bands=3)
    return base.new_from_image([255, 255, 255]).copy(interpretation="srgb")


def _embed_diff_page(image: pyvips.Image, width: int, height: int) -> pyvips.Image:
    if image.width == width and image.height == height:
        return image
    # 画像サイズ差によって存在しない右端・下端は白紙として比較する
    return image.embed(
        0,
        0,
        width,
        height,
        extend="background",
        background=[255, 255, 255],
    )


def _make_diff_vips_page(old_image: pyvips.Image, new_image: pyvips.Image) -> pyvips.Image:
    # RGBのいずれかが異なる画素だけを差分対象にする
    changed = (old_image != new_image).bandbool("or")
    old_ink = (255 - old_image).bandmean()
    new_ink = (255 - new_image).bandmean()

    # 白地に対して濃くなった画素を追加、薄くなった画素を削除と判定する
    added = changed & (new_ink >= old_ink)
    deleted = changed & (new_ink < old_ink)
    # 未変更画素は新画像をそのまま使い、黒は黒、白は白で残す
    result = added.ifthenelse(
        [255, 0, 0],
        deleted.ifthenelse([0, 0, 255], new_image),
    )
    return result.cast("uchar").copy(interpretation="srgb")


def _write_diff_pages_tiff(diff_pages, temp_path: Path, progress_callback=None):
    """ページ固有サイズを保った差分TIFFを一時ファイルへ書き込む。"""
    page_count = len(diff_pages)
    first_page = diff_pages[0]
    has_variable_sizes = any(
        page.width != first_page.width or page.height != first_page.height
        for page in diff_pages[1:]
    )

    if not has_variable_sizes:
        # 同一サイズならlibvipsのストリーミング書き込みを維持する
        joined = pyvips.Image.arrayjoin(diff_pages, across=1)
        joined.set_type(pyvips.GValue.gint_type, "page-height", first_page.height)

        def on_vips_eval(_image, progress):
            write_percent = 30 + int(max(0, min(100, int(progress.percent))) * 0.65)
            _report_diff_progress(progress_callback, write_percent, "差分TIFFを書き込んでいます")

        joined.set_progress(True)
        joined.signal_connect("eval", on_vips_eval)
        _report_diff_progress(progress_callback, 30, "差分TIFFを書き込んでいます")
        joined.write_to_file(str(temp_path), compression=DIFF_TIFF_COMPRESSION)
        _report_diff_progress(progress_callback, 95, "差分TIFFの書き込みが完了しました")
        return

    # libvipsの複数ページ保存は全ページを同じ寸法に揃えるため、ページごとに
    # 一度保存し、Pillowで可変サイズのマルチページTIFFへまとめる
    page_dir = temp_path.with_name(f"{temp_path.name}.pages")
    page_dir.mkdir()
    page_paths = []
    pil_pages = []
    try:
        _report_diff_progress(progress_callback, 30, "差分TIFFを書き込んでいます")
        for page_index, page in enumerate(diff_pages):
            page_path = page_dir / f"{page_index:06d}.tif"
            page.write_to_file(str(page_path), compression=DIFF_TIFF_COMPRESSION)
            page_paths.append(page_path)
            page_percent = 30 + int(45 * (page_index + 1) / page_count)
            _report_diff_progress(progress_callback, page_percent, "差分TIFFを書き込んでいます")

        for page_path in page_paths:
            pil_pages.append(PILImage.open(page_path))
        pil_pages[0].save(
            str(temp_path),
            save_all=True,
            append_images=pil_pages[1:],
            compression="tiff_lzw",
        )
        _report_diff_progress(progress_callback, 95, "差分TIFFの書き込みが完了しました")
    finally:
        for page in pil_pages:
            page.close()
        try:
            shutil.rmtree(page_dir)
        except OSError:
            log_exception("Diff TIFF page temp cleanup failed temp=%s", str(page_dir))


def _is_file_sharing_violation(error: OSError) -> bool:
    # Windowsの共有違反はPython上でwinerror=32、errno=13として通知される
    winerror = getattr(error, "winerror", None)
    if winerror is not None:
        return winerror == 32
    return isinstance(error, PermissionError) and getattr(error, "errno", None) in {13, 32}


def _diff_file_retry_delay_seconds(attempt: int) -> float:
    delay_ms = min(
        DIFF_FILE_RETRY_BASE_DELAY_MS * (2 ** max(0, attempt - 1)),
        DIFF_FILE_RETRY_MAX_DELAY_MS,
    )
    return delay_ms / 1000.0


def _report_diff_progress(progress_callback, percent: int, stage: str):
    if progress_callback is None:
        return
    try:
        progress_callback(max(0, min(100, int(percent))), str(stage))
    except Exception:
        # 進捗表示側の終了などが差分ファイル生成を失敗させないようにする
        log_exception("Diff progress callback failed percent=%s stage=%s", percent, stage)


def _replace_diff_file_with_retry(temp_path: Path, target_path: Path, progress_callback=None):
    for attempt in range(1, DIFF_FILE_REPLACE_MAX_ATTEMPTS + 1):
        try:
            os.replace(str(temp_path), str(target_path))
            if attempt > 1:
                log_info(
                    "Diff TIFF replace recovered temp=%s target=%s attempts=%s",
                    str(temp_path),
                    str(target_path),
                    attempt,
                )
            return
        except OSError as e:
            can_retry = _is_file_sharing_violation(e) and attempt < DIFF_FILE_REPLACE_MAX_ATTEMPTS
            if not can_retry:
                log_exception(
                    "Diff TIFF replace failed temp=%s target=%s attempts=%s winerror=%s errno=%s target_exists=%s temp_exists=%s",
                    str(temp_path),
                    str(target_path),
                    attempt,
                    getattr(e, "winerror", None),
                    getattr(e, "errno", None),
                    target_path.exists(),
                    temp_path.exists(),
                )
                raise

            delay_seconds = _diff_file_retry_delay_seconds(attempt)
            log_info(
                "Diff TIFF replace sharing_violation retry=%s/%s delay_ms=%s temp=%s target=%s",
                attempt,
                DIFF_FILE_REPLACE_MAX_ATTEMPTS,
                int(delay_seconds * 1000),
                str(temp_path),
                str(target_path),
            )
            _report_diff_progress(
                progress_callback,
                96,
                f"保存ファイルを使用中のため再試行しています ({attempt}/{DIFF_FILE_REPLACE_MAX_ATTEMPTS})",
            )
            # libvipsが書き込み直後のファイルを保持する場合に備えてキャッシュと参照を解放する
            _drop_vips_caches()
            gc.collect()
            time.sleep(delay_seconds)


def _create_diff_tiff(
    old_file_path: str,
    new_file_path: str,
    output_path: str,
    progress_callback=None,
) -> Path:
    """新旧TIFFの全ページ差分をLZW圧縮のマルチページTIFFへ保存する。"""
    _report_diff_progress(progress_callback, 0, "差分画像を準備しています")
    old_count = _vips_tiff_page_count(old_file_path)
    new_count = _vips_tiff_page_count(new_file_path)
    page_count = max(old_count, new_count)
    _report_diff_progress(progress_callback, 5, f"ページ情報を確認しました ({page_count}ページ)")
    log_info(
        "Diff TIFF source probed old=%s old_pages=%s new=%s new_pages=%s output=%s",
        old_file_path,
        old_count,
        new_file_path,
        new_count,
        output_path,
    )

    old_pages = []
    new_pages = []
    for page_index in range(page_count):
        old_page = None
        new_page = None
        if page_index < old_count:
            old_page = _normalize_diff_vips_image(pyvips.Image.new_from_file(
                old_file_path,
                access="sequential",
                sequential=True,
                autorotate=True,
                page=page_index,
                n=1,
            ))
        if page_index < new_count:
            new_page = _normalize_diff_vips_image(pyvips.Image.new_from_file(
                new_file_path,
                access="sequential",
                sequential=True,
                autorotate=True,
                page=page_index,
                n=1,
            ))
        old_pages.append(old_page)
        new_pages.append(new_page)
        prepared_percent = 5 + int(15 * (page_index + 1) / page_count)
        _report_diff_progress(
            progress_callback,
            prepared_percent,
            f"ページを準備しています ({page_index + 1}/{page_count})",
        )

    diff_pages = []
    for page_index, (old_page, new_page) in enumerate(zip(old_pages, new_pages)):
        # 比較対象のページ内だけでキャンバスを決め、他ページの寸法に影響させない
        canvas_width = max(
            1,
            old_page.width if old_page is not None else 0,
            new_page.width if new_page is not None else 0,
        )
        canvas_height = max(
            1,
            old_page.height if old_page is not None else 0,
            new_page.height if new_page is not None else 0,
        )
        if old_page is None:
            old_page = _white_diff_vips_image(canvas_width, canvas_height)
        else:
            old_page = _embed_diff_page(old_page, canvas_width, canvas_height)
        if new_page is None:
            new_page = _white_diff_vips_image(canvas_width, canvas_height)
        else:
            new_page = _embed_diff_page(new_page, canvas_width, canvas_height)
        diff_pages.append(_make_diff_vips_page(old_page, new_page))
        diff_percent = 20 + int(10 * (page_index + 1) / page_count)
        _report_diff_progress(
            progress_callback,
            diff_percent,
            f"差分を作成しています ({page_index + 1}/{page_count})",
        )

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(
        f".{target_path.stem}.{os.getpid()}.{time.time_ns()}.tmp{target_path.suffix}"
    )
    page_widths = [page.width for page in diff_pages]
    page_heights = [page.height for page in diff_pages]
    log_info(
        "Diff TIFF save prepared target=%s target_exists=%s temp=%s pages=%s page_size_range=%sx%s..%sx%s compression=%s",
        str(target_path),
        target_path.exists(),
        str(temp_path),
        page_count,
        min(page_widths),
        min(page_heights),
        max(page_widths),
        max(page_heights),
        DIFF_TIFF_COMPRESSION,
    )
    try:
        _write_diff_pages_tiff(diff_pages, temp_path, progress_callback)
        temp_size = temp_path.stat().st_size if temp_path.exists() else -1
        log_info("Diff TIFF temp written temp=%s size_bytes=%s", str(temp_path), temp_size)

        # 書き込みグラフを破棄してから置換し、一時TIFFのファイルハンドルを解放する
        diff_pages.clear()
        old_pages.clear()
        new_pages.clear()
        old_page = None
        new_page = None
        _drop_vips_caches()
        gc.collect()

        log_info(
            "Diff TIFF replace start temp=%s target=%s target_exists=%s",
            str(temp_path),
            str(target_path),
            target_path.exists(),
        )
        _report_diff_progress(progress_callback, 96, "保存ファイルを確定しています")
        _replace_diff_file_with_retry(temp_path, target_path, progress_callback)
        log_info("Diff TIFF replace finished target=%s", str(target_path))
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
                log_info("Diff TIFF temp removed temp=%s", str(temp_path))
        except OSError:
            log_exception("Diff TIFF temp cleanup failed temp=%s", str(temp_path))
    _report_diff_progress(progress_callback, 100, "差分ファイルを保存しました")
    return target_path


def _read_tiff_first_value(data: bytes, endian: str, field_type: int):
    unpack_map = {
        1: "B",    # BYTE
        3: "H",    # SHORT
        4: "I",    # LONG
        6: "b",    # SBYTE
        8: "h",    # SSHORT
        9: "i",    # SLONG
        16: "Q",   # LONG8
        17: "q",   # SLONG8
        18: "Q",   # IFD8
    }
    fmt = unpack_map.get(field_type)
    if fmt is None or len(data) < struct.calcsize(fmt):
        return None
    return struct.unpack(endian + fmt, data[:struct.calcsize(fmt)])[0]


def _read_tiff_ifd_tags(file_path: str, page_index: int, tag_ids: set):
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
            if len(header) < 8:
                return {}

            byte_order = header[:2]
            if byte_order == b"II":
                endian = "<"
            elif byte_order == b"MM":
                endian = ">"
            else:
                return {}

            version = struct.unpack(endian + "H", header[2:4])[0]
            if version == 42:
                is_bigtiff = False
                value_field_size = 4
                ifd_offset = struct.unpack(endian + "I", header[4:8])[0]
            elif version == 43 and len(header) >= 16:
                is_bigtiff = True
                value_field_size = 8
                offset_size = struct.unpack(endian + "H", header[4:6])[0]
                if offset_size != 8:
                    return {}
                ifd_offset = struct.unpack(endian + "Q", header[8:16])[0]
            else:
                return {}

            page = 0
            while ifd_offset:
                f.seek(ifd_offset)
                if is_bigtiff:
                    raw_count = f.read(8)
                    if len(raw_count) < 8:
                        return {}
                    entry_count = struct.unpack(endian + "Q", raw_count)[0]
                    entry_size = 20
                    count_fmt = "Q"
                    next_fmt = "Q"
                else:
                    raw_count = f.read(2)
                    if len(raw_count) < 2:
                        return {}
                    entry_count = struct.unpack(endian + "H", raw_count)[0]
                    entry_size = 12
                    count_fmt = "I"
                    next_fmt = "I"

                tags = {}
                for _ in range(entry_count):
                    entry = f.read(entry_size)
                    if len(entry) < entry_size:
                        return {}

                    tag_id = struct.unpack(endian + "H", entry[0:2])[0]
                    if tag_id not in tag_ids:
                        continue

                    field_type = struct.unpack(endian + "H", entry[2:4])[0]
                    value_count = struct.unpack(endian + count_fmt, entry[4:4 + value_field_size])[0]
                    value_or_offset = entry[4 + value_field_size:4 + value_field_size * 2]
                    type_size = {
                        1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8,
                        11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8,
                    }.get(field_type)
                    if not type_size or value_count <= 0:
                        continue

                    byte_count = value_count * type_size
                    if byte_count <= value_field_size:
                        value_bytes = value_or_offset[:byte_count]
                    else:
                        value_offset = struct.unpack(endian + ("Q" if is_bigtiff else "I"), value_or_offset)[0]
                        current = f.tell()
                        f.seek(value_offset)
                        value_bytes = f.read(type_size)
                        f.seek(current)

                    tags[tag_id] = _read_tiff_first_value(value_bytes, endian, field_type)

                next_bytes = f.read(value_field_size)
                if len(next_bytes) < value_field_size:
                    return {}
                next_ifd_offset = struct.unpack(endian + next_fmt, next_bytes)[0]

                if page == page_index:
                    return tags
                page += 1
                ifd_offset = next_ifd_offset
    except OSError:
        return {}
    except (struct.error, OverflowError, ValueError):
        return {}

    return {}


def _tiff_save_options_like_source(file_path: str, page_index: int) -> dict:
    if Path(file_path).suffix.lower() not in {".tif", ".tiff"}:
        return {}

    tags = _read_tiff_ifd_tags(file_path, page_index, {TIFF_TAG_COMPRESSION, TIFF_TAG_BITS_PER_SAMPLE})
    compression_tag = tags.get(TIFF_TAG_COMPRESSION)
    compression = TIFF_COMPRESSION_TO_VIPS.get(compression_tag)
    if not compression:
        return {}

    options = {"compression": compression}
    # Group4圧縮は1bit画像として保存しないとlibtiffがエラーにする
    if compression == "ccittfax4" and tags.get(TIFF_TAG_BITS_PER_SAMPLE) == 1:
        options["bitdepth"] = 1
    return options


def _safe_process_cwd() -> Path:
    candidates = []

    exe_path = Path(sys.executable).resolve()
    if exe_path.exists():
        candidates.append(exe_path.parent)

    script_dir = Path(__file__).resolve().parent
    candidates.append(script_dir)

    candidates.append(Path.home())

    for p in candidates:
        try:
            if p.exists() and p.is_dir():
                return p
        except OSError:
            continue
    return Path.home()


def _pin_process_cwd(reason: str) -> bool:
    try:
        before = Path.cwd()
    except OSError:
        before = None

    target = _safe_process_cwd()
    try:
        os.chdir(target)
    except OSError as e:
        log_info(
            "process_cwd pin failed reason=%s target=%s err=%s",
            reason,
            str(target),
            e,
        )
        return False

    try:
        after = Path.cwd()
    except OSError:
        after = target

    if before != after:
        log_info("process_cwd updated reason=%s before=%s after=%s", reason, str(before), str(after))
    else:
        log_debug("process_cwd unchanged reason=%s cwd=%s", reason, str(after))
    return True


def _vips_error_text(exc: Exception) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _set_png_clipboard_mime_data(mime: QMimeData, data: QByteArray):
    # Office系アプリがWindowsクリップボード形式名"PNG"として取得できるよう設定する
    if sys.platform == "win32":
        mime.setData('application/x-qt-windows-mime;value="PNG"', data)


def _is_black_and_white_tiff(file_path: str, page_index: int) -> bool:
    if Path(file_path).suffix.lower() not in {".tif", ".tiff"}:
        return False
    tags = _read_tiff_ifd_tags(file_path, page_index, {TIFF_TAG_BITS_PER_SAMPLE})
    return tags.get(TIFF_TAG_BITS_PER_SAMPLE) == 1


def _png_clipboard_buffer(vips_img: pyvips.Image, is_black_and_white_tiff: bool):
    if is_black_and_white_tiff:
        # 白黒TIFFは1bit PNGとして出力し、クリップボード転送時のサイズを抑える。
        return vips_img.write_to_buffer(".png", compression=9, bitdepth=1)
    return vips_img.write_to_buffer(".png", compression=9)


def _drop_vips_caches():
    dropped = False
    try:
        drop_fn = getattr(pyvips, "cache_drop_all", None)
        if callable(drop_fn):
            drop_fn()
            dropped = True
        else:
            vop = getattr(pyvips, "voperation", None)
            drop_fn = getattr(vop, "cache_drop_all", None) if vop is not None else None
            if callable(drop_fn):
                drop_fn()
                dropped = True
    except Exception as e:
        log_debug("pyvips cache_drop_all failed err=%s", e)
    return dropped


def _move_window_center_to_cursor(window):
    if window is None:
        return

    cursor_pos = QCursor.pos()
    screen = QGuiApplication.screenAt(cursor_pos)
    if screen is None:
        screen = window.screen()
    if screen is None:
        screen = QApplication.primaryScreen()
    if screen is None:
        return

    available = screen.availableGeometry()
    if available.isNull():
        return

    frame = window.frameGeometry()
    geom = window.geometry()

    frame_w = frame.width()
    frame_h = frame.height()
    if frame_w <= 0 or frame_h <= 0:
        frame_w = max(1, geom.width())
        frame_h = max(1, geom.height())

    frame_offset_x = frame.x() - geom.x()
    frame_offset_y = frame.y() - geom.y()

    target_frame_x = cursor_pos.x() - (frame_w // 2)
    target_frame_y = cursor_pos.y() - (frame_h // 2)

    min_frame_x = available.x()
    min_frame_y = available.y()
    max_frame_x = available.x() + max(0, available.width() - frame_w)
    max_frame_y = available.y() + max(0, available.height() - frame_h)

    clamped_frame_x = max(min_frame_x, min(target_frame_x, max_frame_x))
    clamped_frame_y = max(min_frame_y, min(target_frame_y, max_frame_y))

    window.move(clamped_frame_x - frame_offset_x, clamped_frame_y - frame_offset_y)


def _select_spread_layout_windows(windows, reference_window, screen_geometry: QRect):
    # 見開き配置は、押下ウィンドウと同じモニター内に中心点がある表示中ウィンドウから選ぶ
    ordered_windows = list(windows)
    order_map = {id(w): i for i, w in enumerate(ordered_windows)}
    candidates = []
    for w in ordered_windows:
        if w is None or (not w.isVisible()) or w.isMinimized():
            continue
        center = w.frameGeometry().center()
        if screen_geometry.contains(center):
            candidates.append(w)

    if len(candidates) < 2:
        return []
    if len(candidates) == 2:
        selected = candidates
    else:
        if reference_window not in candidates:
            return []
        reference_center = reference_window.frameGeometry().center()
        others = [w for w in candidates if w is not reference_window]

        def nearest_key(w):
            center = w.frameGeometry().center()
            dx = center.x() - reference_center.x()
            dy = center.y() - reference_center.y()
            return (dx * dx + dy * dy, abs(dx), order_map.get(id(w), len(order_map)))

        selected = [reference_window, min(others, key=nearest_key)]

    # 現在位置の左右関係を保って、左寄りのウィンドウを左配置にする
    return sorted(
        selected,
        key=lambda w: (w.frameGeometry().center().x(), order_map.get(id(w), len(order_map))),
    )


def _select_nearest_image_window(windows, reference_window):
    """表示中かつ画像を開いているウィンドウから操作元に最も近いものを返す。"""
    if reference_window is None:
        return None

    try:
        reference_center = reference_window.frameGeometry().center()
    except RuntimeError:
        return None

    nearest = None
    nearest_key = None
    for order, window in enumerate(windows):
        if window is None or window is reference_window:
            continue
        try:
            if (not window.isVisible()) or window.isMinimized() or not window.view.file_path():
                continue
            center = window.frameGeometry().center()
        except (AttributeError, RuntimeError):
            continue

        dx = center.x() - reference_center.x()
        dy = center.y() - reference_center.y()
        key = (dx * dx + dy * dy, abs(dx), order)
        if nearest_key is None or key < nearest_key:
            nearest = window
            nearest_key = key
    return nearest


def _client_geometry_for_target_frame(window, target_frame_rect: QRect) -> QRect:
    frame = window.frameGeometry()
    geometry = window.geometry()

    left_margin = geometry.left() - frame.left()
    top_margin = geometry.top() - frame.top()
    right_margin = frame.right() - geometry.right()
    bottom_margin = frame.bottom() - geometry.bottom()

    # setGeometryはタイトルバーなどの外枠ではなくクライアント領域を指定するため、外枠ぶんを内側へ寄せる
    width = max(1, target_frame_rect.width() - left_margin - right_margin)
    height = max(1, target_frame_rect.height() - top_margin - bottom_margin)
    return QRect(
        target_frame_rect.left() + left_margin,
        target_frame_rect.top() + top_margin,
        width,
        height,
    )


def _set_window_frame_geometry(window, target_frame_rect: QRect):
    window.setGeometry(_client_geometry_for_target_frame(window, target_frame_rect))


def _scale_to_fit(source_size: QSize, max_size: QSize) -> QSize:
    if not source_size.isValid():
        return QSize()
    if max_size is None or max_size.isEmpty() or not max_size.isValid():
        return QSize(source_size)

    target = source_size.scaled(max_size, Qt.KeepAspectRatio)
    if not target.isValid() or target.width() <= 0 or target.height() <= 0:
        return QSize(source_size)
    return target.boundedTo(source_size)


def _spread_sync_zoom_ratio_changed(current_ratio: float, target_ratio: float) -> bool:
    # 浮動小数点誤差を除外し、パンだけの同期更新をズーム操作と誤判定しない
    current = max(1.0, float(current_ratio))
    target = max(1.0, float(target_ratio))
    tolerance = max(current, target) * max(0.0, float(SPREAD_SYNC_ZOOM_RATIO_EPSILON))
    return abs(target - current) > tolerance


def _decode_vips_image(vips_img: pyvips.Image, max_size: QSize = QSize()):
    # 先に表示サイズ近傍へ縮小してからQImage化し、初期表示の体感速度を上げる
    source_size = QSize(vips_img.width, vips_img.height)
    target_size = _scale_to_fit(source_size, max_size)

    decode_img = vips_img
    if target_size.isValid() and target_size != source_size and source_size.width() > 0 and source_size.height() > 0:
        xscale = target_size.width() / source_size.width()
        yscale = target_size.height() / source_size.height()
        decode_img = decode_img.resize(xscale, vscale=yscale, kernel="linear")

    qimg = _qimage_from_vips(decode_img)
    if qimg.isNull():
        return QImage(), source_size, "pyvips to QImage conversion failed"
    return qimg, source_size, ""


def _load_vips_page(file_path: str, page_index: int, max_size: QSize = QSize()):
    try:
        vips_img = pyvips.Image.new_from_file(
            file_path,
            access="sequential",
            sequential=True,
            autorotate=True,
            page=page_index,
            n=1,
        )
    except pyvips.Error as e:
        return QImage(), QSize(), _vips_error_text(e)

    return _decode_vips_image(vips_img, max_size)


def _qimage_from_vips(vips_img: pyvips.Image) -> QImage:
    # libvipsのバンド構成をQtで扱える8bit RGB/RGBA/Grayへ正規化する
    img = vips_img
    if img.format != "uchar":
        img = img.cast("uchar")

    if img.bands == 1:
        mem = img.write_to_memory()
        qimg = QImage(mem, img.width, img.height, img.width, QImage.Format_Grayscale8)
        return qimg.copy()

    if str(img.interpretation) not in {"srgb", "rgb16", "b-w"}:
        try:
            img = img.colourspace("srgb")
        except pyvips.Error:
            pass

    if img.format != "uchar":
        img = img.cast("uchar")

    if img.bands == 2:
        gray = img.extract_band(0)
        alpha = img.extract_band(1)
        img = gray.bandjoin(gray).bandjoin(gray).bandjoin(alpha)
    elif img.bands >= 4:
        img = img.extract_band(0, n=4)
    elif img.bands == 3:
        pass
    else:
        single = img.extract_band(0)
        mem = single.write_to_memory()
        qimg = QImage(mem, single.width, single.height, single.width, QImage.Format_Grayscale8)
        return qimg.copy()

    mem = img.write_to_memory()
    if img.bands == 3:
        qimg = QImage(mem, img.width, img.height, img.width * 3, QImage.Format_RGB888)
    else:
        qimg = QImage(mem, img.width, img.height, img.width * 4, QImage.Format_RGBA8888)
    return qimg.copy()


# ---------- QGraphicsItem that draws QImage (avoid QPixmap conversion cost) ----------
class ImageItem(QGraphicsItem):
    def __init__(self):

        super().__init__()

        self._img = QImage()
        self._mipmap_levels = {}

    def set_image(self, img: QImage, mipmap_levels=None):
        self.prepareGeometryChange()
        self._img = img
        if img.isNull():
            self._mipmap_levels = {}
        else:
            self._mipmap_levels = {0: img}
            if mipmap_levels:
                for level, mip in mipmap_levels.items():
                    if level > 0 and not mip.isNull():
                        self._mipmap_levels[level] = mip
        self.update()

    def boundingRect(self) -> QRectF:
        if self._img.isNull():
            return QRectF()
        return QRectF(0, 0, self._img.width(), self._img.height())

    def paint(self, painter: QPainter, option, widget=None):
        if self._img.isNull():
            return

        # 縮小率に応じた mipmap を使って 1px 線の欠けを抑える
        world = painter.worldTransform()
        scale = min(abs(world.m11()), abs(world.m22()))
        level = self._required_mipmap_level(scale)
        self._ensure_mipmap_level(level)

        draw_img = self._mipmap_levels.get(level, self._img)

        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, scale < 0.7 or level > 0)
        if draw_img.size() == self._img.size():
            painter.drawImage(0, 0, draw_img)
        else:
            painter.drawImage(
                QRectF(0, 0, self._img.width(), self._img.height()),
                draw_img,
                QRectF(0, 0, draw_img.width(), draw_img.height()),
            )
        painter.restore()

    def _required_mipmap_level(self, scale: float) -> int:
        if scale >= 1.0:
            return 0

        level = 0
        effective_scale = max(scale, 1e-6)
        while effective_scale < 0.4:
            level += 1
            effective_scale *= 2.0
        return level

    def export_mipmap_levels(self):
        return dict(self._mipmap_levels)

    def _ensure_mipmap_level(self, level: int):
        if self._img.isNull() or level <= 0:
            return

        if not self._mipmap_levels:
            self._mipmap_levels = {0: self._img}
        if level in self._mipmap_levels:
            return

        src_level = max((lv for lv in self._mipmap_levels.keys() if lv < level), default=0)
        src_img = self._mipmap_levels.get(src_level, self._img)
        ratio = 1 << max(1, level - src_level)
        target_w = max(1, src_img.width() // ratio)
        target_h = max(1, src_img.height() // ratio)

        if target_w == src_img.width() and target_h == src_img.height():
            self._mipmap_levels[level] = src_img
            return

        self._mipmap_levels[level] = src_img.scaled(
            target_w,
            target_h,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        )


def _crop_handle_rects(rect: QRectF, handle_size: float) -> dict:
    rect = rect.normalized()
    if rect.isNull():
        return {}

    h = min(max(1.0, handle_size), max(1.0, rect.width()), max(1.0, rect.height()))
    half = h / 2.0
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    cx = rect.center().x()
    cy = rect.center().y()

    return {
        "top_left": QRectF(left, top, h, h),
        "top": QRectF(cx - half, top, h, h),
        "top_right": QRectF(right - h, top, h, h),
        "right": QRectF(right - h, cy - half, h, h),
        "bottom_right": QRectF(right - h, bottom - h, h, h),
        "bottom": QRectF(cx - half, bottom - h, h, h),
        "bottom_left": QRectF(left, bottom - h, h, h),
        "left": QRectF(left, cy - half, h, h),
    }


# ---------- Worker: sequentially decode ALL pages once ----------
class AllPagesLoadSignals(QObject):
    loaded = Signal(int, QImage, str, int)   # (index, image, error, generation)
    page_count = Signal(int, int)            # (total pages, generation)
    finished = Signal(str, int)              # (error or "", generation)


class AllPagesLoadTask(QRunnable):
    def __init__(self, file_path: str, generation: int, preview_max_size=None):
        super().__init__()
        self.file_path = file_path
        self.generation = generation
        self.preview_max_size = QSize(preview_max_size) if preview_max_size else QSize()
        self._cancel_requested = False
        self.signals = AllPagesLoadSignals()

    def cancel(self):
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    def run(self):
        t0 = time.perf_counter()
        # プレビュー用の低解像度を全ページ先読み（ページ移動の待ちを減らす）
        log_info(
            "AllPagesLoadTask(py vips) start file=%s generation=%s preview_max=%s",
            self.file_path,
            self.generation,
            _size_text(self.preview_max_size),
        )
        if self._is_cancelled():
            log_info("AllPagesLoadTask canceled before start generation=%s", self.generation)
            self.signals.finished.emit("cancelled", self.generation)
            return

        try:
            probe = pyvips.Image.new_from_file(
                self.file_path,
                access="sequential",
                sequential=True,
                autorotate=True,
                page=0,
                n=1,
            )
        except pyvips.Error as e:
            err = _vips_error_text(e)
            log_info("AllPagesLoadTask header failed generation=%s err=%s", self.generation, err)
            self.signals.finished.emit(err, self.generation)
            return

        count = 1
        if probe.get_typeof("n-pages") != 0:
            try:
                count = int(probe.get("n-pages"))
            except Exception:
                count = 1
        if count <= 0:
            count = 1
        log_info("AllPagesLoadTask page_count=%s generation=%s", count, self.generation)
        self.signals.page_count.emit(count, self.generation)

        # page 0
        if self._is_cancelled():
            log_info("AllPagesLoadTask canceled before page0 generation=%s", self.generation)
            self.signals.finished.emit("cancelled", self.generation)
            return

        p0_t = time.perf_counter()
        img0, _, err = _decode_vips_image(probe, self.preview_max_size)
        if img0.isNull() or err:
            if not err:
                err = "page 0 decode failed"
            log_info("AllPagesLoadTask page=0 failed generation=%s err=%s", self.generation, err)
            self.signals.finished.emit(err, self.generation)
            return
        log_debug(
            "AllPagesLoadTask page=0 decoded generation=%s img=%s elapsed_ms=%.1f",
            self.generation,
            _img_text(img0),
            (time.perf_counter() - p0_t) * 1000.0,
        )
        self.signals.loaded.emit(0, img0, "", self.generation)

        # pages 1..N-1 sequentially
        for i in range(1, count):
            if self._is_cancelled():
                log_info("AllPagesLoadTask canceled in loop page=%s generation=%s", i, self.generation)
                self.signals.finished.emit("cancelled", self.generation)
                return

            pi_t = time.perf_counter()
            img, _, err = _load_vips_page(self.file_path, i, self.preview_max_size)
            if img.isNull() or err:
                if not err:
                    err = "decode failed"
                log_info("AllPagesLoadTask page=%s failed generation=%s err=%s", i, self.generation, err)
                self.signals.loaded.emit(i, QImage(), err, self.generation)
                continue
            log_debug(
                "AllPagesLoadTask page=%s decoded generation=%s img=%s elapsed_ms=%.1f",
                i,
                self.generation,
                _img_text(img),
                (time.perf_counter() - pi_t) * 1000.0,
            )
            self.signals.loaded.emit(i, img, "", self.generation)

        log_info("AllPagesLoadTask finished generation=%s elapsed_ms=%.1f", self.generation, (time.perf_counter() - t0) * 1000.0)
        self.signals.finished.emit("", self.generation)


class FullResPageSignals(QObject):
    loaded = Signal(int, QImage, str, int, QSize)  # (page, image, error, generation, source_size)


class FullResPageTask(QRunnable):
    def __init__(self, file_path: str, page_index: int, generation: int, target_size: QSize = QSize()):
        super().__init__()
        self.file_path = file_path
        self.page_index = page_index
        self.generation = generation
        self.target_size = QSize(target_size)
        self.signals = FullResPageSignals()

    def run(self):
        t0 = time.perf_counter()
        log_debug(
            "FullResPageTask(py vips) start page=%s generation=%s target=%s file=%s",
            self.page_index,
            self.generation,
            _size_text(self.target_size),
            self.file_path,
        )
        img, source_size, err = _load_vips_page(self.file_path, self.page_index, self.target_size)
        if img.isNull() or err:
            if not err:
                err = "decode failed"
            log_info(
                "FullResPageTask read failed page=%s generation=%s err=%s elapsed_ms=%.1f",
                self.page_index,
                self.generation,
                err,
                (time.perf_counter() - t0) * 1000.0,
            )
            self.signals.loaded.emit(self.page_index, QImage(), err, self.generation, source_size)
            return

        log_debug(
            "FullResPageTask decoded page=%s generation=%s img=%s elapsed_ms=%.1f",
            self.page_index,
            self.generation,
            _img_text(img),
            (time.perf_counter() - t0) * 1000.0,
        )
        self.signals.loaded.emit(self.page_index, img, "", self.generation, source_size)


class DiffTiffSignals(QObject):
    progress = Signal(int, str)          # (進捗率, 処理内容)
    finished = Signal(bool, str, str)  # (成功, 保存先, エラー)


class DiffTiffTask(QRunnable):
    def __init__(self, old_file_path: str, new_file_path: str, output_path: str):
        super().__init__()
        self.old_file_path = old_file_path
        self.new_file_path = new_file_path
        self.output_path = output_path
        self.signals = DiffTiffSignals()
        self._last_progress = -1
        self._last_progress_stage = ""

    def _emit_progress(self, percent: int, stage: str):
        percent = max(self._last_progress, max(0, min(100, int(percent))))
        stage = str(stage)
        # libvipsのeval通知は高頻度なため、同一内容のSignalは送らない
        if percent == self._last_progress and stage == self._last_progress_stage:
            return
        self._last_progress = percent
        self._last_progress_stage = stage
        self.signals.progress.emit(percent, stage)

    def run(self):
        started = time.perf_counter()
        log_info(
            "DiffTiffTask start old=%s new=%s output=%s",
            self.old_file_path,
            self.new_file_path,
            self.output_path,
        )
        try:
            saved_path = _create_diff_tiff(
                self.old_file_path,
                self.new_file_path,
                self.output_path,
                self._emit_progress,
            )
        except Exception as e:
            error = _vips_error_text(e)
            log_exception(
                "DiffTiffTask failed old=%s new=%s output=%s err=%s",
                self.old_file_path,
                self.new_file_path,
                self.output_path,
                error,
            )
            self.signals.finished.emit(False, self.output_path, error)
            return

        log_info(
            "DiffTiffTask finished output=%s elapsed_ms=%.1f",
            str(saved_path),
            (time.perf_counter() - started) * 1000.0,
        )
        self.signals.finished.emit(True, str(saved_path), "")


# ---------- View ----------
class ImageView(QGraphicsView):
    state_changed = Signal()
    file_dropped = Signal(str)
    crop_save_requested = Signal()
    close_requested = Signal()
    window_maximize_toggle_requested = Signal()
    page_step_requested = Signal(int)
    synchronized_view_changed = Signal(object)

    def __init__(self, parent=None):

        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._item = ImageItem()
        self._scene.addItem(self._item)

        # pan/zoom（あなたの環境では問題ないとのことなので最小）
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.viewport().setMouseTracking(True)
        self._fit_mode = True

        # 見開き同期は有効時だけ表示状態を収集し、通常表示への負荷を避ける
        self._view_sync_enabled = False
        self._applying_synchronized_view = False
        self._view_sync_apply_generation = 0
        self._view_sync_timer = QTimer(self)
        self._view_sync_timer.setSingleShot(True)
        self._view_sync_timer.timeout.connect(self._emit_synchronized_view_state)

        # document state
        self._file_path = ""
        self._page_index = 0
        self._page_count = 0
        self._requested_page = 0

        # caches
        self._image_cache = OrderedDict()  # page -> QImage (LRU)
        self.max_image_cache_pages = 64    # 10ページ程度なら余裕でOK
        self._fullres_cache = OrderedDict()  # page -> QImage (LRU full-res)
        self.max_fullres_cache_pages = 4
        self._page_source_sizes = {}  # page -> QSize(original source size)
        self._mipmap_cache = OrderedDict()  # page -> {level: QImage}
        self.max_mipmap_cache_pages = 16

        # トリミング操作状態
        self._crop_view_rect = QRectF()
        self._right_crop_start_view = None
        self._right_crop_started = False
        self._right_press_view_pos = None
        self._left_crop_mode = ""
        self._left_crop_start_rect = QRectF()
        self._left_press_view_pos = None

        # loading progress
        self._loaded_pages = set()
        self._last_error = ""

        # worker pool
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(max(1, min(2, QThreadPool.globalInstance().maxThreadCount())))
        self._fullres_pool = QThreadPool()
        self._fullres_pool.setMaxThreadCount(1)

        self._current_task = None
        self._load_generation = 0
        self._fullres_pending_pages = set()
        self._deferred_fullres_request = False
        self._zoom_interacting = False
        self._idle_fullres_timer = QTimer(self)
        self._idle_fullres_timer.setSingleShot(True)
        self._idle_fullres_timer.timeout.connect(self._request_fullres_current_page)
        self._zoom_interaction_timer = QTimer(self)
        self._zoom_interaction_timer.setSingleShot(True)
        self._zoom_interaction_timer.timeout.connect(self._on_zoom_interaction_idle)
        self._preview_resize_timer = QTimer(self)
        self._preview_resize_timer.setSingleShot(True)
        self._preview_resize_timer.timeout.connect(self._refresh_current_preview_size)
        log_info(
            "ImageView init pool_threads=%s fullres_threads=%s",
            self._pool.maxThreadCount(),
            self._fullres_pool.maxThreadCount(),
        )

    def _stop_background_tasks(
        self,
        reason: str,
        wait_for_done: bool,
        wait_timeout_ms: int = WORKER_DRAIN_TIMEOUT_MS,
    ):
        if self._current_task is not None:
            self._current_task.cancel()
            self._current_task = None
            log_debug("ImageView stop_background_tasks cancel preload reason=%s", reason)

        self._idle_fullres_timer.stop()
        self._zoom_interaction_timer.stop()
        self._preview_resize_timer.stop()
        self._fullres_pending_pages.clear()
        self._deferred_fullres_request = False
        self._zoom_interacting = False

        self._pool.clear()
        self._fullres_pool.clear()

        if wait_for_done:
            preload_done = self._pool.waitForDone(wait_timeout_ms)
            fullres_done = self._fullres_pool.waitForDone(wait_timeout_ms)
            if preload_done is False or fullres_done is False:
                log_info(
                    "ImageView stop_background_tasks timeout reason=%s preload_done=%s fullres_done=%s",
                    reason,
                    preload_done,
                    fullres_done,
                )

        dropped = _drop_vips_caches()
        if dropped:
            gc.collect()
            log_debug("ImageView stop_background_tasks dropped_vips_cache reason=%s", reason)

    def file_path(self):
        return self._file_path

    def page_index(self):
        return self._page_index

    def page_count(self):
        return self._page_count

    def loaded_count(self):
        return len(self._loaded_pages)

    def last_error(self):
        return self._last_error

    def has_image(self):
        return not self._item.boundingRect().isNull()

    def is_fit_mode(self) -> bool:
        return self._fit_mode

    def set_view_sync_enabled(self, enabled: bool):
        # 通常モードではタイマーも状態計算も動かさない
        self._view_sync_enabled = bool(enabled)
        if not self._view_sync_enabled:
            self._view_sync_apply_generation += 1
            self._view_sync_timer.stop()

    def synchronized_view_state(self):
        br = self._item.boundingRect()
        if br.isNull() or br.width() <= 0.0 or br.height() <= 0.0:
            return None

        center = self.mapToScene(self.viewport().rect().center())
        center_x = (center.x() - br.left()) / br.width()
        center_y = (center.y() - br.top()) / br.height()
        fit_scale = max(1e-9, self._synchronized_fit_scale_for_viewport())
        zoom_ratio = max(1.0, self._current_view_scale() / fit_scale)
        return {
            "fit_mode": bool(self._fit_mode),
            "zoom_ratio": zoom_ratio,
            "center_x": max(0.0, min(1.0, center_x)),
            "center_y": max(0.0, min(1.0, center_y)),
        }

    def _schedule_synchronized_view_state(self):
        # 連続パン時の通知を約60fpsに抑えて操作感を維持する
        if (
            not self._view_sync_enabled
            or self._applying_synchronized_view
            or self._view_sync_timer.isActive()
        ):
            return
        self._view_sync_timer.start(max(1, int(SPREAD_SYNC_INTERVAL_MS)))

    def _emit_synchronized_view_state(self):
        if not self._view_sync_enabled or self._applying_synchronized_view:
            return
        state = self.synchronized_view_state()
        if state is not None:
            self.synchronized_view_changed.emit(state)

    def apply_synchronized_view_state(self, state) -> bool:
        if not isinstance(state, dict) or not self.has_image():
            return False

        self._view_sync_apply_generation += 1
        apply_generation = self._view_sync_apply_generation
        self._applying_synchronized_view = True
        self._view_sync_timer.stop()
        try:
            if bool(state.get("fit_mode", False)):
                self.fit_in_view()
                return True

            zoom_ratio = max(1.0, float(state.get("zoom_ratio", 1.0)))
            center_x = max(0.0, min(1.0, float(state.get("center_x", 0.5))))
            center_y = max(0.0, min(1.0, float(state.get("center_y", 0.5))))

            fit_scale = max(1e-9, self._synchronized_fit_scale_for_viewport())
            current_zoom_ratio = max(1.0, self._current_view_scale() / fit_scale)
            if _spread_sync_zoom_ratio_changed(current_zoom_ratio, zoom_ratio):
                # 倍率変更時だけプレビューへ切り替え、パン同期中は高解像度表示を維持する
                started_zoom_interaction = self._mark_zoom_interacting()
                if started_zoom_interaction:
                    preview = self._image_cache.get(self._page_index)
                    detail = self._fullres_cache.get(self._page_index)
                    if (
                        preview is not None
                        and not preview.isNull()
                        and detail is not None
                        and not detail.isNull()
                    ):
                        self._show_page(self._page_index, keep_view=True)

            br = self._item.boundingRect()
            if br.isNull() or br.width() <= 0.0 or br.height() <= 0.0:
                return False

            target_scale = self._synchronized_fit_scale_for_viewport() * zoom_ratio
            self.resetTransform()
            self.scale(target_scale, target_scale)
            self.centerOn(
                br.left() + center_x * br.width(),
                br.top() + center_y * br.height(),
            )
            # スクロールバー表示でviewport寸法が変わった後、同じ中心を再適用する
            QTimer.singleShot(
                0,
                lambda generation=apply_generation, x=center_x, y=center_y: (
                    self._finish_synchronized_center(generation, x, y)
                ),
            )
            self._fit_mode = False
            self.state_changed.emit()
            self._schedule_fullres_upgrade(FULLRES_IDLE_DELAY_MS)
            return True
        except (TypeError, ValueError):
            return False
        finally:
            self._applying_synchronized_view = False

    def _finish_synchronized_center(self, generation: int, center_x: float, center_y: float):
        if generation != self._view_sync_apply_generation or not self.has_image():
            return
        br = self._item.boundingRect()
        if br.isNull():
            return
        self.centerOn(
            br.left() + center_x * br.width(),
            br.top() + center_y * br.height(),
        )

    def current_pixel_size_text(self) -> str:
        source_size = self._page_source_sizes.get(self._page_index)
        if source_size is not None and source_size.isValid():
            return f"({source_size.width()},{source_size.height()})"

        img = self._fullres_cache.get(self._page_index)
        if img is not None and not img.isNull():
            return f"({img.width()},{img.height()})"

        img = self._image_cache.get(self._page_index)
        if img is not None and not img.isNull():
            return f"({img.width()},{img.height()})"

        return "(-,-)"

    def crop_selection_rect(self) -> QRectF:
        return self._crop_view_rect_to_scene_rect(self._crop_view_rect)

    def save_crop_to_file(self, output_path: str):
        if not self._file_path:
            return False, "", "トリミング失敗: 画像が開かれていません"

        image_bounds = self._item.boundingRect()
        crop_rect = self.crop_selection_rect()
        if image_bounds.isNull() or not self._crop_rect_is_usable(crop_rect):
            return False, "", "トリミング失敗: 範囲が選択されていません"

        target_path = Path(output_path)
        try:
            if target_path.parent != Path("."):
                target_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return False, str(target_path), f"トリミング失敗: 保存先フォルダを作成できません ({e})"

        try:
            vips_img = pyvips.Image.new_from_file(
                self._file_path,
                access="random",
                autorotate=True,
                page=self._page_index,
                n=1,
            )

            scale_x = vips_img.width / max(1.0, image_bounds.width())
            scale_y = vips_img.height / max(1.0, image_bounds.height())
            left = crop_rect.left() - image_bounds.left()
            top = crop_rect.top() - image_bounds.top()
            right = crop_rect.right() - image_bounds.left()
            bottom = crop_rect.bottom() - image_bounds.top()

            crop_x0 = int(round(left * scale_x))
            crop_y0 = int(round(top * scale_y))
            crop_x1 = int(round(right * scale_x))
            crop_y1 = int(round(bottom * scale_y))
            crop_x0 = min(max(0, crop_x0), max(0, vips_img.width - 1))
            crop_y0 = min(max(0, crop_y0), max(0, vips_img.height - 1))
            crop_x1 = min(max(crop_x0 + 1, crop_x1), vips_img.width)
            crop_y1 = min(max(crop_y0 + 1, crop_y1), vips_img.height)

            cropped = vips_img.crop(crop_x0, crop_y0, crop_x1 - crop_x0, crop_y1 - crop_y0)
            save_options = _tiff_save_options_like_source(self._file_path, self._page_index)
            cropped.write_to_file(str(target_path), **save_options)
            extra_png_path = None
            if save_options.get("compression") == "lzw":
                extra_png_path = target_path.with_suffix(".png")
                cropped.write_to_file(str(extra_png_path))
        except Exception as e:
            return False, str(target_path), f"トリミング失敗: {_vips_error_text(e)}"

        saved_files_text = str(target_path)
        if extra_png_path is not None:
            saved_files_text += f" / {extra_png_path}"

        try:
            self._copy_vips_image_to_clipboard_png(cropped, self._file_path, self._page_index)
        except Exception as e:
            return True, str(target_path), f"保存完了: {saved_files_text}  クリップボードコピー失敗: {_vips_error_text(e)}"

        return True, str(target_path), f"保存完了: {saved_files_text}  クリップボード: PNG"

    def _copy_vips_image_to_clipboard_png(self, vips_img: pyvips.Image, source_file_path: str, page_index: int):
        # 巨大画像でQImage展開のメモリ上限に触れないよう、PNGバイト列をWindows形式"PNG"で載せる
        is_black_and_white = _is_black_and_white_tiff(source_file_path, page_index)
        png_bytes = _png_clipboard_buffer(vips_img, is_black_and_white)
        data = QByteArray(bytes(png_bytes))
        mime = QMimeData()
        _set_png_clipboard_mime_data(mime, data)
        clipboard = QApplication.clipboard()
        if clipboard is None:
            raise RuntimeError("clipboard is unavailable")
        clipboard.setMimeData(mime)

    def _image_bounds(self) -> QRectF:
        return self._item.boundingRect()

    def _image_view_bounds(self) -> QRectF:
        bounds = self._image_bounds()
        if bounds.isNull():
            return QRectF()
        return QRectF(self.mapFromScene(bounds).boundingRect())

    def _crop_view_bounds(self) -> QRectF:
        image_view_bounds = self._image_view_bounds()
        if image_view_bounds.isNull():
            return QRectF()
        return image_view_bounds.intersected(QRectF(self.viewport().rect()))

    def _clamp_view_pos_to_image(self, view_pos: QPointF) -> QPointF:
        bounds = self._crop_view_bounds()
        if bounds.isNull():
            return QPointF(view_pos)
        return QPointF(
            min(max(view_pos.x(), bounds.left()), bounds.right()),
            min(max(view_pos.y(), bounds.top()), bounds.bottom()),
        )

    def _normalize_crop_rect(self, rect: QRectF) -> QRectF:
        bounds = self._crop_view_bounds()
        if bounds.isNull():
            return QRectF()
        return QRectF(rect).normalized().intersected(bounds)

    def _crop_rect_is_usable(self, rect: QRectF) -> bool:
        return (not rect.isNull()) and rect.width() >= CROP_MIN_SIZE_PX and rect.height() >= CROP_MIN_SIZE_PX

    def _sync_crop_viewport_update_mode(self):
        if CROP_FULL_VIEWPORT_UPDATE and not self._crop_view_rect.isNull():
            mode = QGraphicsView.FullViewportUpdate
        else:
            mode = QGraphicsView.SmartViewportUpdate

        if self.viewportUpdateMode() != mode:
            self.setViewportUpdateMode(mode)

    def _set_crop_rect(self, rect: QRectF):
        normalized = self._normalize_crop_rect(rect)
        if normalized.isNull():
            self._crop_view_rect = QRectF()
            self._sync_crop_viewport_update_mode()
            self.viewport().update()
            return
        self._crop_view_rect = normalized
        self._sync_crop_viewport_update_mode()
        self.viewport().update()

    def _clear_crop_selection(self):
        self._crop_view_rect = QRectF()
        self._right_crop_start_view = None
        self._right_crop_started = False
        self._right_press_view_pos = None
        self._left_crop_mode = ""
        self._left_press_view_pos = None
        self.viewport().unsetCursor()
        self._sync_crop_viewport_update_mode()
        self.viewport().update()

    def _crop_view_rect_to_scene_rect(self, view_rect: QRectF) -> QRectF:
        image_bounds = self._image_bounds()
        rect = QRectF(view_rect).normalized()
        if image_bounds.isNull() or rect.isNull():
            return QRectF()

        top_left = self.mapToScene(rect.topLeft().toPoint())
        bottom_right = self.mapToScene(rect.bottomRight().toPoint())
        return QRectF(top_left, bottom_right).normalized().intersected(image_bounds)

    def _crop_hit_test(self, view_pos: QPointF) -> str:
        crop_rect = QRectF(self._crop_view_rect).normalized()
        if crop_rect.isNull():
            return ""

        for name, handle_rect in _crop_handle_rects(crop_rect, CROP_HANDLE_SIZE_PX).items():
            if handle_rect.contains(view_pos):
                return name

        edge = CROP_EDGE_GRAB_WIDTH_PX
        half = edge / 2.0
        left = crop_rect.left()
        right = crop_rect.right()
        top = crop_rect.top()
        bottom = crop_rect.bottom()
        width = crop_rect.width()
        height = crop_rect.height()

        if QRectF(left, top - half, width, edge).contains(view_pos):
            return "top"
        if QRectF(left, bottom - half, width, edge).contains(view_pos):
            return "bottom"
        if QRectF(left - half, top, edge, height).contains(view_pos):
            return "left"
        if QRectF(right - half, top, edge, height).contains(view_pos):
            return "right"
        if crop_rect.contains(view_pos):
            return "inside"
        return ""

    def _cursor_for_crop_hit(self, hit: str):
        if hit in {"top_left", "bottom_right"}:
            return Qt.SizeFDiagCursor
        if hit in {"top_right", "bottom_left"}:
            return Qt.SizeBDiagCursor
        if hit in {"left", "right"}:
            return Qt.SizeHorCursor
        if hit in {"top", "bottom"}:
            return Qt.SizeVerCursor
        if hit == "inside":
            return Qt.PointingHandCursor
        return None

    def _update_crop_cursor(self, view_pos):
        if self._left_crop_mode or self._right_crop_start_view is not None:
            return
        hit = self._crop_hit_test(QPointF(view_pos))
        cursor = self._cursor_for_crop_hit(hit)
        if cursor is None:
            self.viewport().unsetCursor()
        else:
            self.viewport().setCursor(cursor)

    def _mouse_moved_far(self, start_pos, current_pos) -> bool:
        if start_pos is None:
            return False
        distance = (current_pos - start_pos).manhattanLength()
        threshold = max(0, int(CROP_DRAG_START_DISTANCE_PX))
        if threshold <= 0:
            return distance > 0
        return distance >= threshold

    def _move_crop_rect_by_delta(self, start_rect: QRectF, delta: QPointF) -> QRectF:
        rect = QRectF(start_rect).normalized()
        if rect.isNull():
            return QRectF()

        moved = rect.translated(delta)
        bounds = self._crop_view_bounds()
        if bounds.isNull():
            return moved

        # ラバーバンド全体が画像表示範囲内に残るよう、移動後の位置を補正する
        if moved.width() <= bounds.width():
            if moved.left() < bounds.left():
                moved.moveLeft(bounds.left())
            if moved.right() > bounds.right():
                moved.moveRight(bounds.right())
        else:
            moved.moveLeft(bounds.left())

        if moved.height() <= bounds.height():
            if moved.top() < bounds.top():
                moved.moveTop(bounds.top())
            if moved.bottom() > bounds.bottom():
                moved.moveBottom(bounds.bottom())
        else:
            moved.moveTop(bounds.top())

        return moved

    def _adjust_crop_rect(self, mode: str, view_pos: QPointF) -> QRectF:
        pos = self._clamp_view_pos_to_image(view_pos)
        rect = QRectF(self._left_crop_start_rect).normalized()
        left = rect.left()
        right = rect.right()
        top = rect.top()
        bottom = rect.bottom()

        if "left" in mode:
            left = pos.x()
        if "right" in mode:
            right = pos.x()
        if "top" in mode:
            top = pos.y()
        if "bottom" in mode:
            bottom = pos.y()

        return QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()

    def load_file(self, file_path: str) -> bool:
        log_info("ImageView load_file start file=%s", file_path)
        self._load_generation += 1
        self._stop_background_tasks("load_file", wait_for_done=True)
        self._clear_crop_selection()
        self._file_path = file_path
        self._page_index = 0
        self._requested_page = 0
        self._page_count = 0
        self._last_error = ""
        self._loaded_pages.clear()
        self._image_cache.clear()
        self._fullres_cache.clear()
        self._fullres_pending_pages.clear()
        self._deferred_fullres_request = False
        self._zoom_interacting = False
        self._page_source_sizes.clear()
        self._mipmap_cache.clear()
        log_debug(
            "ImageView load_file reset generation=%s preview_target=%s",
            self._load_generation,
            _size_text(self._preview_decode_size()),
        )

        # pyvipsで表示近傍の解像度を先に読む（Fit初期表示を高速化）
        task = AllPagesLoadTask(self._file_path, self._load_generation, self._preview_decode_size())
        task.signals.page_count.connect(self._on_page_count)
        task.signals.loaded.connect(self._on_page_loaded)
        task.signals.finished.connect(self._on_finished)
        self._current_task = task
        self._pool.start(task)

        self.state_changed.emit()
        log_info("ImageView load_file queued generation=%s", self._load_generation)
        return True

    def clear_document(self):
        had_file = bool(self._file_path)
        self._load_generation += 1
        self._stop_background_tasks("clear_document", wait_for_done=True, wait_timeout_ms=-1)
        self._clear_crop_selection()

        self._file_path = ""
        self._page_index = 0
        self._requested_page = 0
        self._page_count = 0
        self._last_error = ""
        self._loaded_pages.clear()
        self._image_cache.clear()
        self._fullres_cache.clear()
        self._fullres_pending_pages.clear()
        self._deferred_fullres_request = False
        self._zoom_interacting = False
        self._page_source_sizes.clear()
        self._mipmap_cache.clear()

        self.resetTransform()
        self._fit_mode = True
        self._item.set_image(QImage())
        self._scene.setSceneRect(self._item.boundingRect())
        self.state_changed.emit()
        log_info("ImageView clear_document done had_file=%s generation=%s", had_file, self._load_generation)

    def set_page(self, index: int) -> bool:
        if not self._file_path:
            log_debug("ImageView set_page ignored (no file) index=%s", index)
            return False
        if self._page_count > 0 and (index < 0 or index >= self._page_count):
            log_debug("ImageView set_page out_of_range index=%s page_count=%s", index, self._page_count)
            return False
        if index != self._page_index:
            self._clear_crop_selection()

        self._requested_page = index
        log_info(
            "ImageView set_page requested=%s current=%s cached=%s loaded=%s/%s",
            index,
            self._page_index,
            index in self._image_cache,
            len(self._loaded_pages),
            self._page_count,
        )

        if index in self._image_cache:
            self._show_page(index)
        else:
            # まだ読み込み中。読み込み完了時に自動表示される。
            self.state_changed.emit()

        self._schedule_preview_resize_update()
        self._schedule_fullres_upgrade(FULLRES_PAGE_CHANGE_DELAY_MS)

        return True

    def next_page(self) -> bool:
        return self.set_page(self._page_index + 1)

    def prev_page(self) -> bool:
        return self.set_page(self._page_index - 1)

    def fit_in_view(self):
        br = self._item.boundingRect()
        if br.isNull():
            return
        self.resetTransform()
        self.fitInView(br, Qt.KeepAspectRatio)
        self._fit_mode = True
        log_debug("ImageView fit_in_view br=%s scale=%.5f", _rect_text(br), self._current_view_scale())
        self.state_changed.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        log_debug(
            "ImageView resize viewport=%sx%s fit_mode=%s has_image=%s",
            self.viewport().width(),
            self.viewport().height(),
            self._fit_mode,
            self.has_image(),
        )
        if self._fit_mode and self.has_image():
            self.fit_in_view()
        self._schedule_preview_resize_update()
        if self.has_image():
            self._schedule_fullres_upgrade(FULLRES_AFTER_RESIZE_DELAY_MS)

    def paintEvent(self, event):
        super().paintEvent(event)
        self._paint_crop_rubber_band()

    def _paint_crop_rubber_band(self):
        rect = QRectF(self._crop_view_rect).normalized()
        if rect.isNull():
            return

        painter = QPainter(self.viewport())
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(QPen(QColor(0, 120, 215), CROP_BORDER_WIDTH_PX, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        handle_pen_width = max(1.0, CROP_BORDER_WIDTH_PX * 0.5)
        painter.setPen(QPen(QColor(0, 80, 160), handle_pen_width))
        painter.setBrush(QBrush(QColor(255, 255, 255, 220)))
        for handle_rect in _crop_handle_rects(rect, CROP_HANDLE_SIZE_PX).values():
            painter.drawRect(handle_rect)
        painter.restore()
        painter.end()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return

        if event.modifiers() & Qt.ControlModifier:
            self._resize_window_with_ctrl_wheel(delta)
            event.accept()
            return

        br = self._item.boundingRect()
        if br.isNull():
            return

        before_scale = self._current_view_scale()
        steps = max(1, abs(delta) // 120)
        zoom_in = delta > 0

        can_change_scale = True
        if not zoom_in:
            if self._fit_mode:
                can_change_scale = False
            else:
                fit_scale_before = self._fit_scale_for_viewport()
                can_change_scale = before_scale > fit_scale_before * 1.001

        if can_change_scale:
            started_zoom_interaction = self._mark_zoom_interacting()
            if started_zoom_interaction:
                preview = self._image_cache.get(self._page_index)
                detail = self._fullres_cache.get(self._page_index)
                if (
                    preview is not None
                    and not preview.isNull()
                    and detail is not None
                    and not detail.isNull()
                ):
                    log_debug(
                        "ImageView wheel switch_to_preview_on_zoom_start index=%s preview=%s detail=%s",
                        self._page_index,
                        _img_text(preview),
                        _img_text(detail),
                    )
                    self._show_page(self._page_index, keep_view=True)

        for _ in range(steps):
            if zoom_in:
                self.scale(1.25, 1.25)
                self._fit_mode = False
                continue

            if self._fit_mode:
                # フィット状態ではこれ以上縮小しない
                self.fit_in_view()
                break

            current_scale = self._current_view_scale()
            fit_scale = self._fit_scale_for_viewport()
            next_scale = current_scale * 0.8
            if next_scale <= fit_scale * 1.001:
                self.fit_in_view()
                break
            self.scale(0.8, 0.8)

        after_scale = self._current_view_scale()
        if (not zoom_in) and (not self._fit_mode):
            fit_scale_after = self._fit_scale_for_viewport()
            snap_ratio = max(1.0, float(FIT_SNAP_TOLERANCE_RATIO))
            if fit_scale_after > 0.0 and after_scale <= fit_scale_after * snap_ratio:
                log_debug(
                    "ImageView wheel snap_to_fit after_scale=%.5f fit_scale=%.5f ratio=%.5f",
                    after_scale,
                    fit_scale_after,
                    snap_ratio,
                )
                self.fit_in_view()
                after_scale = self._current_view_scale()

        scale_changed = abs(after_scale - before_scale) > 1e-6
        log_debug(
            "ImageView wheel delta=%s steps=%s zoom_in=%s can_change=%s scale_changed=%s scale_before=%.5f scale_after=%.5f fit_mode=%s target_decode=%s",
            delta,
            steps,
            zoom_in,
            "yes" if can_change_scale else "no",
            "yes" if scale_changed else "no",
            before_scale,
            after_scale,
            self._fit_mode,
            _size_text(self._target_decode_size_for_current_view()),
        )

        if not scale_changed:
            if (not zoom_in) and self._fit_mode and (not self._zoom_interacting):
                detail = self._fullres_cache.get(self._page_index)
                if detail is not None and not detail.isNull():
                    log_debug(
                        "ImageView wheel keep_detail_in_fit_on_no_scale index=%s detail=%s",
                        self._page_index,
                        _img_text(detail),
                    )
                    self._show_page(self._page_index, keep_view=True)
            event.accept()
            return

        self.state_changed.emit()
        self._schedule_fullres_upgrade(FULLRES_IDLE_DELAY_MS)
        self._schedule_synchronized_view_state()
        event.accept()

    def _resize_window_with_ctrl_wheel(self, delta: int):
        window = self.window()
        if window is None:
            return
        if window.isMaximized() or window.isFullScreen():
            return
        keep_scene_center = None
        if (not self._fit_mode) and self.has_image():
            keep_scene_center = self.mapToScene(self.viewport().rect().center())
        screen = window.screen()
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()

        steps = max(1, abs(delta) // 120)
        zoom_in = delta > 0
        base = max(1.001, float(CTRL_WHEEL_WINDOW_SCALE_BASE))
        factor = (base ** steps) if zoom_in else ((1.0 / base) ** steps)

        old_w = window.width()
        old_h = window.height()
        old_geometry = window.geometry()
        old_frame = window.frameGeometry()
        old_cx = old_frame.center().x()
        old_cy = old_frame.center().y()
        old_view_w = max(1, self.viewport().width())
        old_view_h = max(1, self.viewport().height())
        frame_extra_w = max(0, old_frame.width() - old_geometry.width())
        frame_extra_h = max(0, old_frame.height() - old_geometry.height())
        max_w = max(MIN_WINDOW_SIZE[0], available.width() - frame_extra_w)
        max_h = max(MIN_WINDOW_SIZE[1], available.height() - frame_extra_h)
        chrome_w = max(0, old_w - old_view_w)
        chrome_h = max(0, old_h - old_view_h)
        min_view_w = max(1, MIN_WINDOW_SIZE[0] - chrome_w)
        min_view_h = max(1, MIN_WINDOW_SIZE[1] - chrome_h)
        max_view_w = max(1, max_w - chrome_w)
        max_view_h = max(1, max_h - chrome_h)
        if max_view_w < min_view_w:
            max_view_w = min_view_w
        if max_view_h < min_view_h:
            max_view_h = min_view_h
        min_scale = max(min_view_w / old_view_w, min_view_h / old_view_h)
        max_scale = min(max_view_w / old_view_w, max_view_h / old_view_h)

        mode = "ratio"
        scale_x = factor
        scale_y = factor

        if zoom_in:
            # 片辺が上限に達したら比率固定を外し、残りの辺も画面上限まで拡げられるようにする
            at_limit = old_view_w >= max_view_w or old_view_h >= max_view_h
            reach_limit = (old_view_w * factor) >= max_view_w or (old_view_h * factor) >= max_view_h
            if at_limit or reach_limit:
                mode = "fill"
                new_view_w = int(round(old_view_w * factor))
                new_view_h = int(round(old_view_h * factor))
                new_view_w = min(max_view_w, max(min_view_w, new_view_w))
                new_view_h = min(max_view_h, max(min_view_h, new_view_h))
                scale_x = new_view_w / old_view_w
                scale_y = new_view_h / old_view_h
            else:
                if max_scale <= 1.0:
                    return
                scale = min(factor, max_scale)
                new_view_w = int(round(old_view_w * scale))
                new_view_h = int(round(old_view_h * scale))
                new_view_w = min(max_view_w, max(min_view_w, new_view_w))
                new_view_h = min(max_view_h, max(min_view_h, new_view_h))
                scale_x = scale
                scale_y = scale
        else:
            if min_scale >= 1.0:
                return
            scale = max(factor, min_scale)
            new_view_w = int(round(old_view_w * scale))
            new_view_h = int(round(old_view_h * scale))
            new_view_w = min(max_view_w, max(min_view_w, new_view_w))
            new_view_h = min(max_view_h, max(min_view_h, new_view_h))
            scale_x = scale
            scale_y = scale

        new_w = max(MIN_WINDOW_SIZE[0], new_view_w + chrome_w)
        new_h = max(MIN_WINDOW_SIZE[1], new_view_h + chrome_h)
        new_w = min(max_w, new_w)
        new_h = min(max_h, new_h)
        if new_w == old_w and new_h == old_h:
            return

        window.resize(new_w, new_h)
        new_frame = window.frameGeometry()
        target_x = old_cx - (new_frame.width() // 2)
        target_y = old_cy - (new_frame.height() // 2)
        min_x = available.left()
        min_y = available.top()
        max_x = available.right() - new_frame.width() + 1
        max_y = available.bottom() - new_frame.height() + 1
        if max_x < min_x:
            max_x = min_x
        if max_y < min_y:
            max_y = min_y
        clamped_x = min(max(target_x, min_x), max_x)
        clamped_y = min(max(target_y, min_y), max_y)
        window.move(clamped_x, clamped_y)
        if keep_scene_center is not None:
            self.centerOn(keep_scene_center)
        log_debug(
            "ImageView ctrl+wheel window_resize delta=%s steps=%s zoom_in=%s mode=%s base=%.3f old=%sx%s new=%sx%s view_old=%sx%s view_new=%sx%s scale_xy=%.4f,%.4f center=%s,%s pos=%s,%s keep_scene_center=%s",
            delta,
            steps,
            zoom_in,
            mode,
            base,
            old_w,
            old_h,
            new_w,
            new_h,
            old_view_w,
            old_view_h,
            new_view_w,
            new_view_h,
            scale_x,
            scale_y,
            old_cx,
            old_cy,
            clamped_x,
            clamped_y,
            "yes" if keep_scene_center is not None else "no",
        )

    def mousePressEvent(self, event):
        view_pos = event.position()

        if event.button() == Qt.RightButton:
            # 右クリックは既存のトリミング枠を消し、ドラッグ時だけ新しい枠を作る
            self._clear_crop_selection()
            if self.has_image():
                self._right_crop_start_view = self._clamp_view_pos_to_image(view_pos)
                self._right_press_view_pos = view_pos
                self._right_crop_started = False
                self.viewport().setCursor(Qt.CrossCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton and not self._crop_view_rect.isNull():
            hit = self._crop_hit_test(view_pos)
            if hit and hit != "inside":
                self._left_crop_mode = hit
                self._left_crop_start_rect = QRectF(self._crop_view_rect)
                self._left_press_view_pos = view_pos
                cursor = self._cursor_for_crop_hit(hit)
                if cursor is not None:
                    self.viewport().setCursor(cursor)
                event.accept()
                return
            if hit == "inside":
                self._left_crop_mode = "inside_pending_save"
                self._left_crop_start_rect = QRectF(self._crop_view_rect)
                self._left_press_view_pos = view_pos
                self.viewport().setCursor(Qt.PointingHandCursor)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 画像表示領域の左ダブルクリックでウィンドウ最大化を切り替える
            self.window_maximize_toggle_requested.emit()
            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        view_pos = event.position()

        if self._right_crop_start_view is not None and (event.buttons() & Qt.RightButton):
            if (not self._right_crop_started) and (not self._mouse_moved_far(self._right_press_view_pos, view_pos)):
                event.accept()
                return
            self._right_crop_started = True
            current = self._clamp_view_pos_to_image(view_pos)
            self._set_crop_rect(QRectF(self._right_crop_start_view, current))
            event.accept()
            return

        if self._left_crop_mode:
            if self._left_crop_mode in {"top_left", "top", "top_right", "right", "bottom_right", "bottom", "bottom_left", "left"}:
                self._set_crop_rect(self._adjust_crop_rect(self._left_crop_mode, view_pos))
                event.accept()
                return

            if self._left_crop_mode == "inside_pending_save":
                if self._mouse_moved_far(self._left_press_view_pos, view_pos):
                    self._left_crop_mode = "inside_move"
                    self._set_crop_rect(self._move_crop_rect_by_delta(
                        self._left_crop_start_rect,
                        view_pos - self._left_press_view_pos,
                    ))
                    self.viewport().setCursor(Qt.SizeAllCursor)
                event.accept()
                return

            if self._left_crop_mode == "inside_move":
                self._set_crop_rect(self._move_crop_rect_by_delta(
                    self._left_crop_start_rect,
                    view_pos - self._left_press_view_pos,
                ))
                event.accept()
                return

        super().mouseMoveEvent(event)
        if event.buttons() & Qt.LeftButton:
            self._schedule_synchronized_view_state()
        if event.buttons() == Qt.NoButton:
            self._update_crop_cursor(view_pos)

    def mouseReleaseEvent(self, event):
        view_pos = event.position()

        if event.button() == Qt.RightButton and self._right_crop_start_view is not None:
            if self._right_crop_started:
                self._set_crop_rect(QRectF(self._right_crop_start_view, self._clamp_view_pos_to_image(view_pos)))
                if not self._crop_rect_is_usable(self._crop_view_rect):
                    self._clear_crop_selection()
                else:
                    self._right_crop_start_view = None
                    self._right_press_view_pos = None
                    self._right_crop_started = False
                    self._update_crop_cursor(view_pos)
            else:
                self._clear_crop_selection()
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._left_crop_mode:
            mode = self._left_crop_mode
            should_save = mode == "inside_pending_save" and not self._mouse_moved_far(self._left_press_view_pos, view_pos)
            if mode in {"top_left", "top", "top_right", "right", "bottom_right", "bottom", "bottom_left", "left"}:
                self._set_crop_rect(self._adjust_crop_rect(mode, view_pos))
            self._left_crop_mode = ""
            self._left_press_view_pos = None
            self._left_crop_start_rect = QRectF()
            if should_save:
                self.crop_save_requested.emit()
            self._update_crop_cursor(view_pos)
            event.accept()
            return

        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton:
            self._schedule_fullres_upgrade(FULLRES_IDLE_DELAY_MS)
            if self._view_sync_enabled:
                self._view_sync_timer.stop()
                self._emit_synchronized_view_state()

    def keyPressEvent(self, event):
        key = event.key()
        if _is_plain_key_event(event, APP_CLOSE_SHORTCUT_KEY):
            self.close_requested.emit()
            event.accept()
            return
        if key == Qt.Key_PageDown:
            self.page_step_requested.emit(1)
            event.accept()
            return
        if key == Qt.Key_PageUp:
            self.page_step_requested.emit(-1)
            event.accept()
            return
        super().keyPressEvent(event)

    def _extract_drop_file_path(self, event):
        md = event.mimeData()
        if md is None or not md.hasUrls():
            return ""

        for url in md.urls():
            if not url.isLocalFile():
                continue
            p = url.toLocalFile()
            if p and Path(p).is_file():
                return p
        return ""

    def dragEnterEvent(self, event):
        if self._extract_drop_file_path(event):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if self._extract_drop_file_path(event):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        path = self._extract_drop_file_path(event)
        if path:
            event.acceptProposedAction()
            self.file_dropped.emit(path)
            return
        super().dropEvent(event)

    # ---- internals ----
    def _preview_decode_size(self) -> QSize:
        vp = self.viewport().rect()
        w = vp.width() if vp.width() > 0 else self.width()
        h = vp.height() if vp.height() > 0 else self.height()
        w = max(512, w)
        h = max(512, h)
        return QSize(w, h)

    def _target_decode_size_for_current_view(self) -> QSize:
        br = self._item.boundingRect()
        if br.isNull():
            return QSize()

        scale = self._current_view_scale()
        target_w = max(1, int(round(br.width() * scale)))
        target_h = max(1, int(round(br.height() * scale)))
        target_w = max(256, target_w)
        target_h = max(256, target_h)
        return QSize(target_w, target_h)

    def _schedule_preview_resize_update(self, delay_ms: int = PREVIEW_RESIZE_UPDATE_DELAY_MS):
        if not self._file_path:
            return
        if self._page_index < 0:
            return
        self._preview_resize_timer.start(max(1, int(delay_ms)))

    def _refresh_current_preview_size(self):
        if not self._file_path or self._page_index < 0:
            return
        index = self._page_index
        preview = self._image_cache.get(index)
        if preview is None or preview.isNull():
            return
        detail = self._fullres_cache.get(index)
        if detail is None or detail.isNull():
            return

        viewport_target = self._preview_decode_size()
        target_scale = max(1.0, float(PREVIEW_RESIZE_TARGET_SCALE))
        if target_scale > 1.0:
            viewport_target = QSize(
                max(1, int(round(viewport_target.width() * target_scale))),
                max(1, int(round(viewport_target.height() * target_scale))),
            )
        desired = _scale_to_fit(QSize(detail.width(), detail.height()), viewport_target)
        if not desired.isValid():
            return

        if (
            abs(preview.width() - desired.width()) < PREVIEW_RESIZE_MIN_DELTA_PX
            and abs(preview.height() - desired.height()) < PREVIEW_RESIZE_MIN_DELTA_PX
        ):
            # 差分が小さいときは再生成コストを避ける
            if index == self._page_index:
                self._show_page(index, keep_view=True)
            return

        t0 = time.perf_counter()
        mode = Qt.SmoothTransformation if PREVIEW_RESIZE_SMOOTH else Qt.FastTransformation
        new_preview = detail.scaled(desired, Qt.KeepAspectRatio, mode)
        if new_preview.isNull():
            return

        self._lru_put(index, new_preview)
        self._mipmap_cache.pop(index, None)
        log_info(
            "ImageView preview_resize_update index=%s old=%s new=%s viewport=%s target_scale=%.2f detail=%s elapsed_ms=%.1f",
            index,
            _img_text(preview),
            _img_text(new_preview),
            _size_text(viewport_target),
            target_scale,
            _img_text(detail),
            (time.perf_counter() - t0) * 1000.0,
        )

        if index == self._page_index:
            self._show_page(index, keep_view=True)
        else:
            self.state_changed.emit()

    def _schedule_fullres_upgrade(self, delay_ms: int = FULLRES_IDLE_DELAY_MS):
        if not self._file_path:
            return
        self._idle_fullres_timer.start(max(1, int(delay_ms)))

    def _mark_zoom_interacting(self) -> bool:
        started = False
        if not self._zoom_interacting:
            self._zoom_interacting = True
            started = True
            log_debug("ImageView zoom_interaction start")
        self._zoom_interaction_timer.start(max(1, int(ZOOM_INTERACTION_IDLE_MS)))
        return started

    def _on_zoom_interaction_idle(self):
        if not self._zoom_interacting:
            return
        self._zoom_interacting = False
        log_debug("ImageView zoom_interaction idle")
        if self._file_path and self.has_image() and self._page_index >= 0:
            self._show_page(self._page_index, keep_view=True)
        self._schedule_fullres_upgrade(FULLRES_AFTER_ZOOM_IDLE_DELAY_MS)

    def _lru_put(self, key, value):
        if key in self._image_cache:
            self._image_cache.move_to_end(key)
            self._image_cache[key] = value
        else:
            self._image_cache[key] = value
            self._image_cache.move_to_end(key)
        while len(self._image_cache) > self.max_image_cache_pages:
            old_key, _ = self._image_cache.popitem(last=False)
            self._mipmap_cache.pop(old_key, None)
            self._fullres_cache.pop(old_key, None)
            self._fullres_pending_pages.discard(old_key)
            self._page_source_sizes.pop(old_key, None)

    def _fullres_lru_put(self, key, value):
        if key in self._fullres_cache:
            self._fullres_cache.move_to_end(key)
            self._fullres_cache[key] = value
        else:
            self._fullres_cache[key] = value
            self._fullres_cache.move_to_end(key)
        while len(self._fullres_cache) > self.max_fullres_cache_pages:
            self._fullres_cache.popitem(last=False)

    def _mipmap_lru_put(self, key, value):
        if key in self._mipmap_cache:
            self._mipmap_cache.move_to_end(key)
            self._mipmap_cache[key] = value
        else:
            self._mipmap_cache[key] = value
            self._mipmap_cache.move_to_end(key)
        while len(self._mipmap_cache) > self.max_mipmap_cache_pages:
            self._mipmap_cache.popitem(last=False)

    def _current_view_scale(self) -> float:
        t = self.transform()
        sx = abs(t.m11())
        sy = abs(t.m22())
        if sx <= 0.0 or sy <= 0.0:
            return 1.0
        return min(sx, sy)

    def _fit_scale_for_viewport(self) -> float:
        br = self._item.boundingRect()
        if br.isNull():
            return 1.0

        vp = self.viewport().rect()
        if vp.width() <= 0 or vp.height() <= 0:
            return 1.0
        return min(vp.width() / br.width(), vp.height() / br.height())

    def _synchronized_fit_scale_for_viewport(self) -> float:
        br = self._item.boundingRect()
        if br.isNull():
            return 1.0

        # ズーム中に表示されたスクロールバー分を戻し、倍率の往復ずれを防ぐ
        vp = self.viewport().rect()
        width = vp.width()
        height = vp.height()
        vertical_bar = self.verticalScrollBar()
        horizontal_bar = self.horizontalScrollBar()
        if vertical_bar.isVisible():
            width += vertical_bar.width()
        if horizontal_bar.isVisible():
            height += horizontal_bar.height()
        if width <= 0 or height <= 0:
            return 1.0
        return min(width / br.width(), height / br.height())

    def _choose_image_for_current_view(self, index: int, preview_img: QImage, detail_img: QImage):
        candidates = []
        if not preview_img.isNull():
            candidates.append(preview_img)
        if not detail_img.isNull():
            candidates.append(detail_img)
        if not candidates:
            return QImage()

        if self._zoom_interacting and not preview_img.isNull():
            if not detail_img.isNull():
                log_debug(
                    "ImageView choose_image prefer_preview_during_zoom index=%s preview=%s detail=%s",
                    index,
                    _img_text(preview_img),
                    _img_text(detail_img),
                )
            return preview_img

        source_size = self._page_source_sizes.get(index)
        if (
            source_size is not None
            and source_size.isValid()
            and not preview_img.isNull()
            and not detail_img.isNull()
        ):
            detail_is_source = detail_img.width() >= source_size.width() and detail_img.height() >= source_size.height()
            if detail_is_source:
                log_debug(
                    "ImageView choose_image prefer_source_detail index=%s preview=%s detail=%s source=%s",
                    index,
                    _img_text(preview_img),
                    _img_text(detail_img),
                    _size_text(source_size),
                )
                return detail_img

        # Fit表示では detail を優先して初期表示品質を揃える
        if self._fit_mode and not detail_img.isNull():
            log_debug(
                "ImageView choose_image prefer_detail_in_fit index=%s preview=%s detail=%s",
                index,
                _img_text(preview_img),
                _img_text(detail_img),
            )
            return detail_img

        target = self._target_decode_size_for_current_view()
        if not target.isValid():
            return candidates[0]

        covering = [
            img for img in candidates
            if img.width() >= target.width() and img.height() >= target.height()
        ]
        if covering:
            return min(covering, key=lambda img: img.width() * img.height())
        return max(candidates, key=lambda img: img.width() * img.height())

    def _show_page(self, index: int, keep_view: bool = False):
        preview_img = self._image_cache.get(index, QImage())
        detail_img = self._fullres_cache.get(index, QImage())
        img = self._choose_image_for_current_view(index, preview_img, detail_img)
        if img is None or img.isNull():
            log_debug(
                "ImageView _show_page skipped index=%s keep_view=%s preview=%s detail=%s",
                index,
                keep_view,
                _img_text(preview_img),
                _img_text(detail_img),
            )
            return

        old_br = self._item.boundingRect()
        old_center = self.mapToScene(self.viewport().rect().center()) if keep_view and not old_br.isNull() else None
        old_page = self._page_index
        if index != old_page:
            self._clear_crop_selection()

        # 直前ページの縮小キャッシュを退避
        if not self._item.boundingRect().isNull() and self._page_index in self._image_cache:
            self._mipmap_lru_put(self._page_index, self._item.export_mipmap_levels())

        page_mipmaps = self._mipmap_cache.get(index)
        if page_mipmaps:
            base = page_mipmaps.get(0)
            if base is not None and not base.isNull() and base.size() != img.size():
                page_mipmaps = None
                self._mipmap_cache.pop(index, None)
        if page_mipmaps is not None:
            self._mipmap_cache.move_to_end(index)
        self._item.set_image(img, page_mipmaps)
        self._scene.setSceneRect(self._item.boundingRect())
        self._page_index = index
        new_br = self._item.boundingRect()
        log_info(
            "ImageView _show_page index=%s keep_view=%s fit_mode=%s selected=%s preview=%s detail=%s br_old=%s br_new=%s scale=%.5f",
            index,
            keep_view,
            self._fit_mode,
            _img_text(img),
            _img_text(preview_img),
            _img_text(detail_img),
            _rect_text(old_br),
            _rect_text(new_br),
            self._current_view_scale(),
        )

        if keep_view and (not self._fit_mode) and old_center is not None:
            same_size = (
                abs(old_br.width() - new_br.width()) < 1e-6
                and abs(old_br.height() - new_br.height()) < 1e-6
            )
            if same_size:
                log_debug("ImageView _show_page keep_view skip_adjust same_size old=%s new=%s", _rect_text(old_br), _rect_text(new_br))
            elif old_br.width() > 0 and old_br.height() > 0 and new_br.width() > 0 and new_br.height() > 0:
                norm_x = old_center.x() / old_br.width()
                norm_y = old_center.y() / old_br.height()
                t = self.transform()
                t.scale(old_br.width() / new_br.width(), old_br.height() / new_br.height())
                self.setTransform(t)
                self.centerOn(norm_x * new_br.width(), norm_y * new_br.height())
                log_debug(
                    "ImageView _show_page keep_view adjust norm=(%.6f, %.6f) scale_after=%.5f",
                    norm_x,
                    norm_y,
                    self._current_view_scale(),
                )
        else:
            self.fit_in_view()

        self.setFocus(Qt.OtherFocusReason)
        self.state_changed.emit()

    def _request_fullres_current_page(self):
        if not self._file_path:
            log_debug("ImageView _request_fullres skip no file")
            return
        if self._zoom_interacting:
            # ズーム連打中はfullres要求を遅延してUI応答性を優先
            self._deferred_fullres_request = True
            log_debug("ImageView _request_fullres defer while zoom interacting")
            return
        if self._current_task is not None:
            # 全ページ先読み中は同時負荷を避けるため後回し
            self._deferred_fullres_request = True
            log_debug("ImageView _request_fullres defer while preload running")
            return
        index = self._page_index
        if index < 0:
            log_debug("ImageView _request_fullres skip invalid index=%s", index)
            return
        if index in self._fullres_pending_pages:
            log_debug("ImageView _request_fullres skip pending index=%s", index)
            return
        if index not in self._image_cache:
            log_debug("ImageView _request_fullres skip preview_missing index=%s", index)
            return

        source_size = self._page_source_sizes.get(index)
        current_target = self._target_decode_size_for_current_view()
        if IDLE_PARTIAL_FULLRES_ENABLED and current_target.isValid():
            scale = max(1.0, float(IDLE_PARTIAL_FULLRES_SCALE))
            target = QSize(
                max(1, int(round(current_target.width() * scale))),
                max(1, int(round(current_target.height() * scale))),
            )
            req_mode = f"scaled_view_x{scale:.2f}"
        else:
            # 旧挙動: sourceサイズが分かっていればそれを要求、未確定ならフルデコード要求
            if source_size is not None and source_size.isValid():
                target = QSize(source_size)
                req_mode = "source_size"
            else:
                target = QSize()
                req_mode = "source_full"

        if source_size is not None and source_size.isValid() and target.isValid():
            bounded = target.boundedTo(source_size)
            if bounded.width() > 0 and bounded.height() > 0:
                if bounded != target:
                    log_debug(
                        "ImageView _request_fullres clamp_target index=%s mode=%s target=%s source=%s bounded=%s",
                        index,
                        req_mode,
                        _size_text(target),
                        _size_text(source_size),
                        _size_text(bounded),
                    )
                target = bounded

        cached = self._fullres_cache.get(index)
        if cached is not None and not cached.isNull():
            if source_size is not None and source_size.isValid():
                if cached.width() >= source_size.width() and cached.height() >= source_size.height():
                    self._deferred_fullres_request = False
                    log_debug(
                        "ImageView _request_fullres skip reached_source index=%s cached=%s source=%s",
                        index,
                        _img_text(cached),
                        _size_text(source_size),
                    )
                    return
            if target.isValid():
                if cached.width() >= target.width() and cached.height() >= target.height():
                    self._deferred_fullres_request = False
                    log_debug(
                        "ImageView _request_fullres skip cached_enough index=%s mode=%s cached=%s target=%s",
                        index,
                        req_mode,
                        _img_text(cached),
                        _size_text(target),
                    )
                    return

        gen = self._load_generation
        task = FullResPageTask(self._file_path, index, gen, target)
        task.signals.loaded.connect(self._on_fullres_loaded)
        self._fullres_pending_pages.add(index)
        self._deferred_fullres_request = False
        log_info(
            "ImageView _request_fullres start index=%s generation=%s mode=%s target=%s cached=%s scale=%.5f",
            index,
            gen,
            req_mode,
            _size_text(target),
            _img_text(cached) if cached is not None else "none",
            self._current_view_scale(),
        )
        self._fullres_pool.start(task)

    @Slot(int, int)
    def _on_page_count(self, cnt: int, generation: int):
        # 世代不一致＝古いタスクの結果なので破棄
        if generation != self._load_generation:
            log_debug(
                "ImageView page_count stale count=%s generation=%s current_generation=%s",
                cnt,
                generation,
                self._load_generation,
            )
            return

        self._page_count = cnt
        log_info("ImageView page_count=%s generation=%s", cnt, generation)
        self.state_changed.emit()

    @Slot(int, QImage, str, int)
    def _on_page_loaded(self, index: int, img: QImage, err: str, generation: int):
        # 世代不一致＝古いタスクの結果なので破棄
        if generation != self._load_generation:
            log_debug(
                "ImageView page_loaded stale index=%s generation=%s current_generation=%s",
                index,
                generation,
                self._load_generation,
            )
            return

        if err:
            self._last_error = f"Page {index+1}: {err}"
            log_info("ImageView page_loaded error index=%s generation=%s err=%s", index, generation, err)
            self.state_changed.emit()
            return

        self._loaded_pages.add(index)
        self._lru_put(index, img)
        log_debug(
            "ImageView page_loaded index=%s generation=%s img=%s loaded=%s/%s requested=%s",
            index,
            generation,
            _img_text(img),
            len(self._loaded_pages),
            self._page_count,
            self._requested_page,
        )

        # 要求中ページが来たら即表示
        if index == self._requested_page:
            self._show_page(index)
        else:
            self.state_changed.emit()

    @Slot(int, QImage, str, int, QSize)
    def _on_fullres_loaded(self, index: int, img: QImage, err: str, generation: int, source_size: QSize):
        self._fullres_pending_pages.discard(index)
        # 世代不一致＝古いタスクの結果なので破棄
        if generation != self._load_generation:
            log_debug(
                "ImageView fullres_loaded stale index=%s generation=%s current_generation=%s",
                index,
                generation,
                self._load_generation,
            )
            return
        if err:
            self._last_error = f"Page {index+1}: {err}"
            log_info("ImageView fullres_loaded error index=%s generation=%s err=%s", index, generation, err)
            self.state_changed.emit()
            return
        if img.isNull():
            log_info("ImageView fullres_loaded null index=%s generation=%s", index, generation)
            return

        if source_size.isValid():
            self._page_source_sizes[index] = QSize(source_size)
        self._fullres_lru_put(index, img)
        self._mipmap_cache.pop(index, None)
        log_info(
            "ImageView fullres_loaded ok index=%s generation=%s img=%s source=%s current_page=%s",
            index,
            generation,
            _img_text(img),
            _size_text(source_size),
            self._page_index,
        )

        if index == self._page_index:
            self._show_page(index, keep_view=True)
            self._schedule_preview_resize_update(1)

    @Slot(str, int)
    def _on_finished(self, err: str, generation: int):
        # 世代不一致＝古いタスクの完了通知なので破棄
        if generation != self._load_generation:
            log_debug(
                "ImageView all_pages_finished stale generation=%s current_generation=%s err=%s",
                generation,
                self._load_generation,
                err,
            )
            return
        self._current_task = None
        if err == "cancelled":
            log_info("ImageView all_pages_finished cancelled generation=%s", generation)
            self.state_changed.emit()
            return

        if err:
            self._last_error = err
            log_info("ImageView all_pages_finished generation=%s err=%s", generation, err)
        else:
            log_info("ImageView all_pages_finished generation=%s ok", generation)
        self.state_changed.emit()
        if not err and self._file_path:
            if self._deferred_fullres_request:
                log_debug("ImageView all_pages_finished run deferred fullres request")
            else:
                log_debug("ImageView all_pages_finished schedule initial fullres request")
            self._schedule_fullres_upgrade(FULLRES_AFTER_PRELOAD_DELAY_MS)


# ---------- MainWindow ----------
class MainWindow(QMainWindow):
    IMAGE_EXTENSIONS = {".tif", ".tiff"}
    new_window_requested = Signal(str)
    spread_layout_requested = Signal()
    diff_requested = Signal()
    diff_file_saved = Signal(str)
    sync_command_requested = Signal(str, int)
    view_sync_requested = Signal(object)

    def __init__(self, enable_tray: bool = True):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(MIN_WINDOW_SIZE[0], MIN_WINDOW_SIZE[1])
        self._apply_window_icon()
        log_info("MainWindow init")

        self.view = ImageView(self)
        self.setCentralWidget(self.view)
        self.view.state_changed.connect(self._update_ui)
        self.view.file_dropped.connect(self._open_dropped_file)
        self.view.crop_save_requested.connect(self._save_current_crop)
        self.view.close_requested.connect(self._close_from_shortcut)
        self.view.window_maximize_toggle_requested.connect(self._toggle_window_maximized)
        self.view.page_step_requested.connect(self._request_page_step)
        self.view.synchronized_view_changed.connect(self.view_sync_requested.emit)
        self._allow_close = False
        self._tray_enabled = bool(enable_tray)
        self._tray_available = False
        self._tray_icon = None
        self._tray_menu = None
        self._tray_close_notice_shown = False
        self._spread_sync_enabled = False
        self._spread_sync_partner = None
        self._diff_pool = QThreadPool(self)
        self._diff_pool.setMaxThreadCount(1)
        self._diff_task = None
        self._diff_progress_percent = 0
        self._diff_progress_stage = ""

        self.act_new_window = QAction("NewWindow", self)
        self.act_new_window.setShortcut("Ctrl+Shift+N")
        self.act_new_window.triggered.connect(self._request_new_window)

        self.act_spread_layout = QAction("Spread(S)", self)
        self.act_spread_layout.setShortcut(Qt.Key_S)
        self.act_spread_layout.setCheckable(True)
        self.act_spread_layout.triggered.connect(self._request_spread_layout)

        self.act_diff = QAction("Diff(D)", self)
        self.act_diff.setShortcut(DIFF_SHORTCUT_KEY)
        self.act_diff.triggered.connect(self._request_diff)

        self.act_save = QAction("Save", self)
        self.act_save.triggered.connect(self._save_current_file_to_desktop)

        self.act_open = QAction("Open", self)
        self.act_open.triggered.connect(self.open_file)

        self.act_prev = QAction("PageUp", self)
        self.act_prev.setShortcut(Qt.Key_PageUp)
        self.act_prev.triggered.connect(lambda _checked=False: self._request_page_step(-1))

        self.act_next = QAction("PageDown", self)
        self.act_next.setShortcut(Qt.Key_PageDown)
        self.act_next.triggered.connect(lambda _checked=False: self._request_page_step(1))

        self.act_fit = QAction("Fit(F)", self)
        self.act_fit.setShortcut(Qt.Key_F)
        self.act_fit.triggered.connect(lambda _checked=False: self._request_sync_command("fit"))

        self.act_next_file = QAction("NextFile(N)", self)
        self.act_next_file.setShortcut(Qt.Key_N)
        self.act_next_file.triggered.connect(lambda _checked=False: self._request_sync_command("file", 1))

        self.act_prev_file = QAction("PrevFile(B)", self)
        self.act_prev_file.setShortcut(Qt.Key_B)
        self.act_prev_file.triggered.connect(lambda _checked=False: self._request_sync_command("file", -1))

        self.act_quit = QAction("Exit", self)
        self.act_quit.triggered.connect(self._quit_from_tray)

        self.crop_save_label = QLabel("　Crop save as : ", self)
        self.crop_save_edit = QLineEdit(_default_crop_save_text(), self)
        self.crop_save_edit.setMinimumWidth(CROP_SAVE_TEXT_MIN_WIDTH)

        self.diff_progress_bar = QProgressBar(self)
        self.diff_progress_bar.setRange(0, 100)
        self.diff_progress_bar.setFixedWidth(DIFF_PROGRESS_BAR_WIDTH_PX)
        self.diff_progress_bar.setFormat("%p%")
        self.diff_progress_bar.setTextVisible(True)
        self.diff_progress_bar.hide()
        self.statusBar().addPermanentWidget(self.diff_progress_bar)

        if self._tray_enabled:
            self._setup_tray_icon()

        tb = self.addToolBar("Main")
        # 指定順: Save, Open, NewWindow, Spread, Diff, Fit, PageUp, PageDown, PrevFile, NextFile
        tb.addAction(self.act_save)
        tb.addAction(self.act_open)
        tb.addAction(self.act_new_window)
        tb.addAction(self.act_spread_layout)
        tb.addAction(self.act_diff)
        tb.addAction(self.act_fit)
        tb.addAction(self.act_prev)
        tb.addAction(self.act_next)
        tb.addAction(self.act_prev_file)
        tb.addAction(self.act_next_file)
        tb.addWidget(self.crop_save_label)
        tb.addWidget(self.crop_save_edit)

        self._dir_files = []
        self._dir_file_index = -1

        self.statusBar().showMessage("Open / PgUp/PgDn=Prev/Next / Wheel=Zoom / Drag=Pan / RightDrag=Crop / C=Close")
        self._update_ui()

    def _apply_window_icon(self):
        icon_path = _find_app_icon_path()
        if not icon_path:
            log_debug("MainWindow icon not found")
            return

        icon = QIcon(icon_path)
        if icon.isNull():
            log_info("MainWindow icon load failed path=%s", icon_path)
            return

        self.setWindowIcon(icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)
        log_info("MainWindow icon applied path=%s", icon_path)

    def _setup_tray_icon(self):
        if not self._tray_enabled:
            return
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray_available = False
            log_info("MainWindow tray unavailable")
            return

        tray_icon = QIcon(self.windowIcon())
        if tray_icon.isNull():
            tray_icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        self._tray_icon = QSystemTrayIcon(tray_icon, self)
        self._tray_icon.setToolTip(WINDOW_TITLE)

        menu = QMenu(self)
        menu.addAction(self.act_open)
        menu.addAction(self.act_new_window)
        menu.addSeparator()
        menu.addAction(self.act_quit)
        self._tray_menu = menu
        self._tray_icon.setContextMenu(menu)
        self._tray_icon.activated.connect(self._on_tray_activated)
        self._tray_icon.show()
        self._tray_available = True
        log_info("MainWindow tray ready")

    def _show_main_window(self):
        if self.isMinimized():
            self.setWindowState((self.windowState() & ~Qt.WindowMinimized) | Qt.WindowActive)
        self.show()
        self.raise_()
        self.activateWindow()
        log_debug("MainWindow show_main_window")
        self._update_ui()

    def _hide_to_tray(self, show_notice: bool = False):
        if not self._tray_available or self._tray_icon is None:
            return
        self.hide()
        log_debug("MainWindow hide_to_tray")
        if show_notice and (not self._tray_close_notice_shown) and self._tray_icon.supportsMessages():
            self._tray_icon.showMessage(
                WINDOW_TITLE,
                "Fast TIFF Viewer is running in the system tray.",
                QSystemTrayIcon.Information,
                3000,
            )
            self._tray_close_notice_shown = True
        self._update_ui()

    def _quit_from_tray(self):
        log_info("MainWindow quit requested from tray")
        self._quit_application()

    def _close_from_shortcut(self):
        # 同期相手を先に閉じてから、操作元を通常の閉じる動作へ流す
        log_info("MainWindow close requested from keyboard shortcut")
        self.sync_command_requested.emit("close", 0)
        self.close()

    def set_spread_sync_state(self, enabled: bool, partner=None):
        # 同期OFF中もペアを保持し、次のSpread押下で同じ2画面へ復帰できるようにする
        self._spread_sync_enabled = bool(enabled)
        self._spread_sync_partner = partner
        self.view.set_view_sync_enabled(self._spread_sync_enabled)
        self.act_spread_layout.setChecked(self._spread_sync_enabled)
        self._update_ui()

    def _enabled_spread_sync_partner(self):
        if not self._spread_sync_enabled or self._spread_sync_partner is None:
            return None
        try:
            self._spread_sync_partner.view
        except RuntimeError:
            return None
        return self._spread_sync_partner

    def _can_page_step(self, step: int) -> bool:
        if not self.view.file_path() or self.view.page_count() <= 1:
            return False
        target = self.view.page_index() + (-1 if step < 0 else 1)
        return 0 <= target < self.view.page_count()

    def _can_file_step(self, step: int) -> bool:
        return bool(self._neighbor_file_path(-1 if step < 0 else 1))

    def _apply_sync_command(self, command: str, value: int = 0) -> bool:
        # 相手側からの適用ではSignalを再送せず、同期ループを防ぐ
        if command == "fit":
            if not self.view.has_image():
                return False
            self.view.fit_in_view()
            return True
        if command == "page":
            return self.view.prev_page() if value < 0 else self.view.next_page()
        if command == "file":
            before_path = self.view.file_path()
            if value < 0:
                self.open_prev_file()
            else:
                self.open_next_file()
            return self.view.file_path() != before_path
        if command == "close":
            self.close()
            return True
        return False

    def _request_sync_command(self, command: str, value: int = 0):
        self._apply_sync_command(command, value)
        # 操作元が末尾でも相手側へ必ず転送し、ベストエフォートで進める
        self.sync_command_requested.emit(command, value)

    @Slot(int)
    def _request_page_step(self, step: int):
        self._request_sync_command("page", -1 if step < 0 else 1)

    def _toggle_window_maximized(self):
        # 画像ビューからの操作だけで最大化/通常化を切り替える
        if self.isMaximized():
            self.showNormal()
            log_info("MainWindow restored by image double click")
            return

        self.showMaximized()
        log_info("MainWindow maximized by image double click")

    def _quit_application(self):
        app = QApplication.instance()
        if app is not None:
            # トレイ常駐や複数ウィンドウでも、終了時のcloseEventが拒否しないよう全ウィンドウを許可状態にする
            for widget in app.topLevelWidgets():
                if isinstance(widget, MainWindow):
                    widget._allow_close = True
                    if widget._tray_icon is not None:
                        widget._tray_icon.hide()
            app.quit()
            return

        self._allow_close = True
        self.close()

    def _on_tray_activated(self, reason):
        if reason in {QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick}:
            if self.isVisible():
                self._hide_to_tray()
            else:
                self._show_main_window()

    def start_in_tray(self):
        if (not self._tray_enabled) or (not self._tray_available) or self._tray_icon is None:
            self._show_main_window()
            log_info("MainWindow start_in_tray fallback to show")
            return
        self._hide_to_tray()
        log_info("MainWindow started in tray mode")

    def _update_ui(self):
        has_file = bool(self.view.file_path())
        pc = self.view.page_count()
        pi = self.view.page_index()
        loaded = self.view.loaded_count()
        err = self.view.last_error()
        mode_text = "Fit" if self.view.is_fit_mode() else "Non-Fit"

        partner = self._enabled_spread_sync_partner()
        partner_can_prev_page = partner._can_page_step(-1) if partner is not None else False
        partner_can_next_page = partner._can_page_step(1) if partner is not None else False
        partner_can_prev_file = partner._can_file_step(-1) if partner is not None else False
        partner_can_next_file = partner._can_file_step(1) if partner is not None else False
        partner_has_image = partner.view.has_image() if partner is not None else False

        self.act_prev.setEnabled(self._can_page_step(-1) or partner_can_prev_page)
        self.act_next.setEnabled(self._can_page_step(1) or partner_can_next_page)
        self.act_fit.setEnabled(self.view.has_image() or partner_has_image)
        self.act_save.setEnabled(has_file)
        self.act_diff.setEnabled(has_file and self._diff_task is None)
        self.act_next_file.setEnabled(self._can_file_step(1) or partner_can_next_file)
        self.act_prev_file.setEnabled(self._can_file_step(-1) or partner_can_prev_file)

        if self._diff_task is not None:
            self.statusBar().showMessage(
                f"差分検出中 ({self._diff_progress_percent}%): {self._diff_progress_stage}"
            )
        elif has_file:
            name = Path(self.view.file_path()).name
            pixel_size = self.view.current_pixel_size_text()
            if pc > 0:
                msg = f"{name}  Page {pi+1}/{pc}  Loaded {loaded}/{pc}  Pixel {pixel_size}  Mode {mode_text}"
            else:
                msg = f"{name}  Page {pi+1}  Loaded {loaded}  Pixel {pixel_size}  Mode {mode_text}"
            if err:
                msg += f"   (Error: {err})"
            self.statusBar().showMessage(msg)
        else:
            self.statusBar().showMessage(f"Open / PgUp/PgDn=Prev/Next / Wheel=Zoom / Drag=Pan / RightDrag=Crop / C=Close  Mode {mode_text}")

    def open_file(self):
        if not self.isVisible():
            self._show_main_window()
        try:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open TIFF",
                _default_open_directory(),
                "TIFF Files (*.tif *.tiff)",
            )
        finally:
            _pin_process_cwd("open_file_dialog")
        if not path:
            log_debug("MainWindow open_file canceled")
            return

        log_info("MainWindow open_file selected path=%s", path)
        self._open_path(path)

    def _save_current_file_to_desktop(self):
        source_path = self.view.file_path()
        if not source_path:
            self.statusBar().showMessage("保存失敗: 開いているファイルがありません")
            log_info("MainWindow desktop_save skipped no_file")
            return

        try:
            target_path = _desktop_copy_target_path(source_path)
        except Exception as e:
            msg = f"保存失敗: 保存先を解釈できません ({e})"
            self.statusBar().showMessage(msg)
            log_info("MainWindow desktop_save resolve_failed source=%s err=%s", source_path, e)
            return

        if target_path.exists():
            reply = QMessageBox.question(
                self,
                "Save",
                f"デスクトップに同じファイル名があります。\n上書き保存しますか？\n\n{target_path}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self.statusBar().showMessage(f"保存キャンセル: {target_path}")
                log_info("MainWindow desktop_save canceled source=%s target=%s", source_path, str(target_path))
                return

        try:
            saved_path = _copy_file_to_desktop(source_path)
        except Exception as e:
            msg = f"保存失敗: {e}"
            self.statusBar().showMessage(msg)
            QMessageBox.warning(
                self,
                "Save failed",
                f"デスクトップへの保存に失敗しました。\n\n元ファイル:\n{source_path}\n\n保存先:\n{target_path}\n\n{e}",
            )
            log_info("MainWindow desktop_save failed source=%s target=%s err=%s", source_path, str(target_path), e)
            return

        self.statusBar().showMessage(f"保存完了: {saved_path}")
        log_info("MainWindow desktop_save ok source=%s target=%s", source_path, str(saved_path))

    def _request_new_window(self):
        self.new_window_requested.emit("")
        log_info("MainWindow requested new empty window")

    def _request_spread_layout(self, _checked=False):
        self.spread_layout_requested.emit()
        log_info("MainWindow requested spread layout")

    def _request_diff(self, _checked=False):
        if self._diff_task is not None:
            self.statusBar().showMessage("差分検出は実行中です")
            return
        self.diff_requested.emit()
        log_info("MainWindow requested diff")

    def start_diff(self, old_file_path: str) -> bool:
        new_file_path = self.view.file_path()
        if not new_file_path or not old_file_path:
            self.statusBar().showMessage("差分検出できません: 新旧画像を開いてください")
            return False

        try:
            target_path = _resolve_diff_save_path(new_file_path, self.crop_save_edit.text())
            target_key = _normalized_absolute_path(str(target_path))
            if target_key in {
                _normalized_absolute_path(new_file_path),
                _normalized_absolute_path(old_file_path),
            }:
                raise ValueError("保存先に新画像または旧画像と同じファイルは指定できません")
        except Exception as e:
            self.statusBar().showMessage(f"差分検出失敗: 保存先を解釈できません ({e})")
            log_info("MainWindow diff resolve_failed err=%s", e)
            return False

        log_info(
            "MainWindow diff start old=%s new=%s save_text=%s target=%s target_exists=%s",
            old_file_path,
            new_file_path,
            self.crop_save_edit.text(),
            str(target_path),
            target_path.exists(),
        )
        task = DiffTiffTask(old_file_path, new_file_path, str(target_path))
        task.signals.progress.connect(self._on_diff_progress)
        task.signals.finished.connect(self._on_diff_finished)
        self._diff_task = task
        self._diff_progress_percent = 0
        self._diff_progress_stage = "差分画像を準備しています"
        self.diff_progress_bar.setValue(0)
        self.diff_progress_bar.show()
        self.act_diff.setEnabled(False)
        self.statusBar().showMessage(
            f"差分検出中: 旧={Path(old_file_path).name} / 新={Path(new_file_path).name}"
        )
        self._diff_pool.start(task)
        return True

    @Slot(int, str)
    def _on_diff_progress(self, percent: int, stage: str):
        if self._diff_task is None:
            return
        self._diff_progress_percent = max(0, min(100, int(percent)))
        self._diff_progress_stage = str(stage)
        self.diff_progress_bar.setValue(self._diff_progress_percent)
        self.statusBar().showMessage(
            f"差分検出中 ({self._diff_progress_percent}%): {self._diff_progress_stage}"
        )

    @Slot(bool, str, str)
    def _on_diff_finished(self, ok: bool, saved_path: str, error: str):
        self._diff_task = None
        self.diff_progress_bar.hide()
        self._diff_progress_percent = 0
        self._diff_progress_stage = ""
        self._update_ui()
        if not ok:
            self.statusBar().showMessage(f"差分検出失敗: {error}")
            QMessageBox.warning(
                self,
                "Diff failed",
                f"差分画像の保存に失敗しました。\n\n{saved_path}\n\n{error}\n\nログ:\n{LOG_FILE_PATH}",
            )
            return

        self.statusBar().showMessage(f"差分保存完了: {saved_path}")
        self.diff_file_saved.emit(saved_path)

    @Slot(str)
    def _open_dropped_file(self, path: str):
        log_info("MainWindow drop_open path=%s", path)
        self._open_path(path)

    @Slot()
    def _save_current_crop(self):
        try:
            target_path = _resolve_crop_save_path(self.view.file_path(), self.crop_save_edit.text())
        except Exception as e:
            msg = f"トリミング失敗: 保存先を解釈できません ({e})"
            self.statusBar().showMessage(msg)
            log_info("MainWindow crop_save resolve_failed err=%s", e)
            return

        ok, saved_path, msg = self.view.save_crop_to_file(str(target_path))
        self.statusBar().showMessage(msg)
        if ok:
            log_info("MainWindow crop_save ok path=%s", saved_path)
        else:
            log_info("MainWindow crop_save failed path=%s msg=%s", saved_path, msg)

    def open_from_cli_args(self, args):
        if not args:
            log_debug("MainWindow cli_open no args")
            return

        path = _normalize_input_path(args[0])
        if not path:
            log_debug("MainWindow cli_open empty first arg")
            return

        self._show_main_window()
        log_info("MainWindow cli_open path=%s", path)
        self._open_path(path)

    def open_next_file(self):
        path = self._neighbor_file_path(1)
        if path:
            log_info("MainWindow next_file path=%s", path)
            self._open_path(path)
        else:
            log_debug("MainWindow next_file no neighbor")

    def open_prev_file(self):
        path = self._neighbor_file_path(-1)
        if path:
            log_info("MainWindow prev_file path=%s", path)
            self._open_path(path)
        else:
            log_debug("MainWindow prev_file no neighbor")

    def _rebuild_directory_image_list(self, current_path: str):
        current = Path(current_path)
        try:
            parent_path = os.path.abspath(os.fspath(current.parent))
            current_resolved = os.path.abspath(os.fspath(current))
            files = []
            # ネットワークフォルダではPath.is_file/resolveの件数分アクセスが重いため、scandirの属性情報で一覧化する
            with os.scandir(parent_path) as entries:
                for entry in entries:
                    if not entry.name.lower().endswith((".tif", ".tiff")):
                        continue
                    if entry.is_file():
                        files.append(os.path.join(parent_path, entry.name))
        except OSError:
            self._dir_files = []
            self._dir_file_index = -1
            return

        files.sort(key=lambda p: Path(p).name.casefold())
        self._dir_files = files
        try:
            self._dir_file_index = self._dir_files.index(current_resolved)
        except ValueError:
            self._dir_file_index = -1
        log_debug(
            "MainWindow rebuild_dir_list dir=%s files=%s current_index=%s",
            str(current.parent),
            len(self._dir_files),
            self._dir_file_index,
        )

    def _neighbor_file_path(self, step: int) -> str:
        if not self._dir_files or self._dir_file_index < 0:
            return ""

        target = self._dir_file_index + step
        if 0 <= target < len(self._dir_files):
            return self._dir_files[target]
        return ""

    def _open_path(self, path: str):
        log_info("MainWindow open_path path=%s", path)
        ok = self.view.load_file(path)
        if not ok:
            log_info("MainWindow open_path failed path=%s", path)
            QMessageBox.warning(self, "Open failed", f"Failed to load:\n{path}")
        else:
            self._rebuild_directory_image_list(path)
            log_info("MainWindow open_path success path=%s", path)
        self._update_ui()

    def _clear_loaded_image_state(self):
        had_file = bool(self.view.file_path())
        self.view.clear_document()
        self._dir_files = []
        self._dir_file_index = -1
        self._update_ui()
        log_info("MainWindow clear_loaded_image_state had_file=%s", had_file)

    def closeEvent(self, event):
        if self._allow_close:
            self._clear_loaded_image_state()
            log_info("MainWindow closeEvent accept (allow_close)")
            super().closeEvent(event)
            return

        if self._tray_available and self._tray_icon is not None:
            log_info("MainWindow closeEvent hide to tray")
            event.ignore()
            self._clear_loaded_image_state()
            self._hide_to_tray(show_notice=True)
            return

        log_info("MainWindow closeEvent accept (tray unavailable)")
        self._clear_loaded_image_state()
        super().closeEvent(event)


class AppController(QObject):
    def __init__(self, app):
        super().__init__(app)
        self._app = app
        self._windows = []
        self._primary_window = None
        self._spread_windows = ()
        self._spread_sync_enabled = False
        self._spread_sync_applying = False
        self._instance_server = SingleInstanceServer(self._on_ipc_message, self)

    def start_single_instance_server(self):
        self._instance_server.start()

    def startup(self, cli_args):
        self._primary_window = self._create_window(enable_tray=True)
        self._primary_window.resize(DEFAULT_WINDOW_SIZE[0], DEFAULT_WINDOW_SIZE[1])
        if cli_args:
            _move_window_center_to_cursor(self._primary_window)
            self._primary_window.show()
            log_info(
                "AppController startup primary shown size=%sx%s",
                self._primary_window.width(),
                self._primary_window.height(),
            )
            QTimer.singleShot(0, lambda: self._primary_window.open_from_cli_args(cli_args))
            return

        self._primary_window.start_in_tray()
        log_info("AppController startup primary started in tray")

    def _create_window(self, enable_tray: bool) -> MainWindow:
        w = MainWindow(enable_tray=enable_tray)
        if not enable_tray:
            w.setAttribute(Qt.WA_DeleteOnClose, True)
        w.new_window_requested.connect(self.open_new_window)
        w.spread_layout_requested.connect(lambda win=w: self.arrange_spread_layout(win))
        w.diff_requested.connect(lambda win=w: self.create_diff(win))
        w.diff_file_saved.connect(self.open_new_window)
        w.sync_command_requested.connect(
            lambda command, value, win=w: self._on_spread_sync_command(win, command, value)
        )
        w.view_sync_requested.connect(
            lambda state, win=w: self._on_spread_view_changed(win, state)
        )
        w.view.state_changed.connect(lambda win=w: self._refresh_spread_pair_ui(win))
        w.destroyed.connect(lambda _=None, win=w: self._on_window_destroyed(win))
        self._windows.append(w)
        log_info("AppController window created enable_tray=%s total=%s", enable_tray, len(self._windows))
        return w

    def create_diff(self, reference_window: MainWindow):
        if reference_window is None or not reference_window.view.file_path():
            if reference_window is not None:
                reference_window.statusBar().showMessage("差分検出できません: 新画像が開かれていません")
            return

        window_states = []
        for window in self._windows:
            try:
                window_states.append({
                    "id": id(window),
                    "visible": window.isVisible(),
                    "minimized": window.isMinimized(),
                    "file": window.view.file_path(),
                })
            except RuntimeError:
                window_states.append({"id": id(window), "destroyed": True})
        log_info("AppController diff window_states=%s", window_states)

        old_window = _select_nearest_image_window(self._windows, reference_window)
        if old_window is None:
            reference_window.statusBar().showMessage(
                "差分検出には、表示中の別ウィンドウで旧画像を開く必要があります"
            )
            log_info("AppController diff skipped no_old_window")
            return

        old_file_path = old_window.view.file_path()
        log_info(
            "AppController diff selected old_window=%s old=%s new_window=%s new=%s",
            id(old_window),
            old_file_path,
            id(reference_window),
            reference_window.view.file_path(),
        )
        reference_window.start_diff(old_file_path)

    def _on_window_destroyed(self, window):
        if window in self._spread_windows:
            # destroyed済みのウィンドウは呼び出さず、残った相手側だけ同期解除する
            remaining = [w for w in self._spread_windows if w is not window]
            self._spread_windows = ()
            self._spread_sync_enabled = False
            self._spread_sync_applying = False
            for other in remaining:
                try:
                    other.set_spread_sync_state(False, None)
                except RuntimeError:
                    pass
        for i, w in enumerate(list(self._windows)):
            if w is window:
                self._windows.pop(i)
                break
        log_info("AppController window destroyed total=%s", len(self._windows))

    def _spread_partner(self, window):
        if len(self._spread_windows) != 2 or window not in self._spread_windows:
            return None
        return self._spread_windows[1] if self._spread_windows[0] is window else self._spread_windows[0]

    def _set_spread_sync_enabled(self, enabled: bool):
        if len(self._spread_windows) != 2:
            enabled = False
        self._spread_sync_enabled = bool(enabled)
        for window in self._spread_windows:
            partner = self._spread_partner(window)
            try:
                window.set_spread_sync_state(self._spread_sync_enabled, partner)
            except RuntimeError:
                pass

    def _clear_spread_sync_session(self):
        old_windows = self._spread_windows
        self._spread_windows = ()
        self._spread_sync_enabled = False
        self._spread_sync_applying = False
        for window in old_windows:
            try:
                window.set_spread_sync_state(False, None)
            except RuntimeError:
                pass

    def _refresh_spread_pair_ui(self, changed_window=None):
        if not self._spread_sync_enabled:
            return
        if changed_window is not None and changed_window not in self._spread_windows:
            return
        for window in self._spread_windows:
            try:
                window._update_ui()
            except RuntimeError:
                pass

    def _synchronize_view_from(self, source_window):
        if not self._spread_sync_enabled or self._spread_sync_applying:
            return
        partner = self._spread_partner(source_window)
        if partner is None:
            return
        state = source_window.view.synchronized_view_state()
        if state is None:
            return
        self._on_spread_view_changed(source_window, state)

    def _on_spread_view_changed(self, source_window, state):
        if not self._spread_sync_enabled or self._spread_sync_applying:
            return
        partner = self._spread_partner(source_window)
        if partner is None:
            return

        self._spread_sync_applying = True
        try:
            partner.view.apply_synchronized_view_state(state)
        finally:
            self._spread_sync_applying = False

    def _on_spread_sync_command(self, source_window, command: str, value: int):
        if not self._spread_sync_enabled or self._spread_sync_applying:
            return
        partner = self._spread_partner(source_window)
        if partner is None:
            return

        if command == "close":
            # Close中のdestroyed通知より先にペアを解消し、二重適用を防ぐ
            self._clear_spread_sync_session()
            partner._apply_sync_command("close", 0)
            return

        self._spread_sync_applying = True
        try:
            # Page/Fileは相手が移動できる場合だけ成功し、片側末尾でも他方を止めない
            partner._apply_sync_command(command, value)
        finally:
            self._spread_sync_applying = False
        self._refresh_spread_pair_ui()

    @Slot(str)
    def open_new_window(self, path: str = ""):
        normalized_path = _normalize_input_path(path) if path else ""
        w = self._create_window(enable_tray=False)
        w.resize(DEFAULT_WINDOW_SIZE[0], DEFAULT_WINDOW_SIZE[1])
        _move_window_center_to_cursor(w)
        w.show()
        log_info("AppController open_new_window path=%s", normalized_path if normalized_path else "(none)")
        if normalized_path:
            QTimer.singleShot(0, lambda p=normalized_path, win=w: win.open_from_cli_args([p]))

    def _bring_spread_windows_to_front(self, reference_window: MainWindow):
        # 操作元を最後に前面化し、2画面を連続した最前面のZオーダーへ移動する
        windows = [window for window in self._spread_windows if window is not reference_window]
        if reference_window in self._spread_windows:
            windows.append(reference_window)

        raised_count = 0
        for window in windows:
            try:
                if window.isMinimized():
                    window.showNormal()
                else:
                    window.show()
                window.raise_()
                raised_count += 1
            except RuntimeError:
                continue

        if reference_window in windows:
            try:
                reference_window.activateWindow()
            except RuntimeError:
                pass
        log_info("AppController spread_windows brought_to_front count=%s", raised_count)

    def arrange_spread_layout(self, reference_window: MainWindow):
        if reference_window is None:
            return

        if reference_window in self._spread_windows and len(self._spread_windows) == 2:
            enable_sync = not self._spread_sync_enabled
            self._set_spread_sync_enabled(enable_sync)
            message = "見開き同期を有効にしました" if enable_sync else "見開き同期を解除しました"
            for window in self._spread_windows:
                window.statusBar().showMessage(message)
            if enable_sync:
                QTimer.singleShot(0, lambda win=reference_window: self._synchronize_view_from(win))
            self._bring_spread_windows_to_front(reference_window)
            log_info("AppController spread_sync toggled enabled=%s", enable_sync)
            return

        reference_center = reference_window.frameGeometry().center()
        screen = QGuiApplication.screenAt(reference_center)
        if screen is None:
            screen = reference_window.screen()
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is None:
            reference_window.statusBar().showMessage("見開き配置できません: モニターを取得できません")
            log_info("AppController spread_layout failed no_screen")
            return

        selected = _select_spread_layout_windows(self._windows, reference_window, screen.geometry())
        if len(selected) != 2:
            reference_window.act_spread_layout.setChecked(
                self._spread_sync_enabled and reference_window in self._spread_windows
            )
            reference_window.statusBar().showMessage("見開き配置には同じモニター上に2つのウィンドウが必要です")
            log_info(
                "AppController spread_layout skipped selected=%s screen=%s",
                len(selected),
                _rect_text(QRectF(screen.geometry())),
            )
            return

        available = screen.availableGeometry()
        left_width = available.width() // 2
        right_width = available.width() - left_width
        left_rect = QRect(available.x(), available.y(), left_width, available.height())
        right_rect = QRect(available.x() + left_width, available.y(), right_width, available.height())

        left_window, right_window = selected
        self._clear_spread_sync_session()
        self._spread_windows = (left_window, right_window)
        self._set_spread_sync_enabled(True)
        for window, rect in ((left_window, left_rect), (right_window, right_rect)):
            # 最大化などの状態を解除してから、対象モニターの半分へ配置する
            window.showNormal()
            _set_window_frame_geometry(window, rect)
            QTimer.singleShot(0, window.view.fit_in_view)

        for window in self._spread_windows:
            window.statusBar().showMessage("見開き配置・同期を有効にしました")
        self._bring_spread_windows_to_front(reference_window)
        log_info(
            "AppController spread_layout arranged left=%s right=%s screen=%s available=%s",
            id(left_window),
            id(right_window),
            _rect_text(QRectF(screen.geometry())),
            _rect_text(QRectF(available)),
        )

    def _on_ipc_message(self, message: str):
        if message.startswith("OPEN\t"):
            path = _normalize_input_path(message[5:])
            if path:
                self.open_new_window(path)
            return
        if message == "PING":
            return


if __name__ == "__main__":
    setup_debug_logging()
    log_info("log_file=%s", str(LOG_FILE_PATH))
    log_info("pyvips=%s libvips=%s.%s.%s", pyvips.__version__, pyvips.version(0), pyvips.version(1), pyvips.version(2))
    _pin_process_cwd("process_start")
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    cli_args = sys.argv[1:]
    ipc_message = _build_ipc_message(cli_args)
    if _send_ipc_message(ipc_message):
        log_info("main delegated_to_existing_instance message=%s", ipc_message)
        sys.exit(0)

    controller = AppController(app)
    controller.start_single_instance_server()
    controller.startup(cli_args)
    sys.exit(app.exec())
