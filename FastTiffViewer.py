import sys
import os
import logging
import time
from pathlib import Path
from collections import OrderedDict

from PySide6.QtCore import Qt, QObject, Signal, Slot, QRunnable, QThreadPool, QRectF, QSize, QTimer, QUrl
from PySide6.QtGui import QAction, QIcon, QImage, QPainter
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
    QStyle,
    QSystemTrayIcon,
)
import pyvips


LOG_FILE_PATH = Path(__file__).with_name("fasttiffviewer_debug.log")
LOGGER = logging.getLogger("fasttiffviewer")
ENABLE_DEBUG_LOGGING = os.getenv("TIFFVIEWER_DEBUG_LOG", "0").strip().lower() in {"1", "true", "yes", "on"}
# 上記 "0"を"1"でlogファイル出力

WINDOW_TITLE = "Fast TIFF Viewer v1.2.0"
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
    sock.disconnectFromServer()
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
        handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        ))
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.DEBUG)
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


def _vips_error_text(exc: Exception) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _scale_to_fit(source_size: QSize, max_size: QSize) -> QSize:
    if not source_size.isValid():
        return QSize()
    if max_size is None or max_size.isEmpty() or not max_size.isValid():
        return QSize(source_size)

    target = source_size.scaled(max_size, Qt.KeepAspectRatio)
    if not target.isValid() or target.width() <= 0 or target.height() <= 0:
        return QSize(source_size)
    return target.boundedTo(source_size)


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


# ---------- View ----------
class ImageView(QGraphicsView):
    state_changed = Signal()
    file_dropped = Signal(str)

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
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self._fit_mode = True

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

    def load_file(self, file_path: str) -> bool:
        log_info("ImageView load_file start file=%s", file_path)
        self._load_generation += 1
        if self._current_task is not None:
            self._current_task.cancel()
            log_debug("ImageView load_file cancel previous task")
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
        self._idle_fullres_timer.stop()
        self._zoom_interaction_timer.stop()
        self._preview_resize_timer.stop()
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
        if self._current_task is not None:
            self._current_task.cancel()
            self._current_task = None
            log_debug("ImageView clear_document cancel current preload task")

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
        self._idle_fullres_timer.stop()
        self._zoom_interaction_timer.stop()
        self._preview_resize_timer.stop()

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

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton:
            self._schedule_fullres_upgrade(FULLRES_IDLE_DELAY_MS)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_PageDown and self.next_page():
            event.accept()
            return
        if key == Qt.Key_PageUp and self.prev_page():
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
        self._allow_close = False
        self._tray_enabled = bool(enable_tray)
        self._tray_available = False
        self._tray_icon = None
        self._tray_menu = None
        self._tray_close_notice_shown = False

        self.act_new_window = QAction("NewWindow", self)
        self.act_new_window.setShortcut("Ctrl+Shift+N")
        self.act_new_window.triggered.connect(self._request_new_window)

        self.act_open = QAction("Open", self)
        self.act_open.triggered.connect(self.open_file)

        self.act_open_new = QAction("Open(NewWindow)", self)
        self.act_open_new.setShortcut("Ctrl+Shift+O")
        self.act_open_new.triggered.connect(self.open_file_in_new_window)

        self.act_prev = QAction("PageUp", self)
        self.act_prev.setShortcut(Qt.Key_PageUp)
        self.act_prev.triggered.connect(self.view.prev_page)

        self.act_next = QAction("PageDown", self)
        self.act_next.setShortcut(Qt.Key_PageDown)
        self.act_next.triggered.connect(self.view.next_page)

        self.act_fit = QAction("Fit(F)", self)
        self.act_fit.setShortcut(Qt.Key_F)
        self.act_fit.triggered.connect(self.view.fit_in_view)

        self.act_next_file = QAction("NextFile(N)", self)
        self.act_next_file.setShortcut(Qt.Key_N)
        self.act_next_file.triggered.connect(self.open_next_file)

        self.act_prev_file = QAction("PrevFile(B)", self)
        self.act_prev_file.setShortcut(Qt.Key_B)
        self.act_prev_file.triggered.connect(self.open_prev_file)

        self.act_show_window = QAction("Show", self)
        self.act_show_window.triggered.connect(self._show_main_window)

        self.act_hide_window = QAction("Hide", self)
        self.act_hide_window.triggered.connect(self._hide_to_tray)

        self.act_quit = QAction("Exit", self)
        self.act_quit.triggered.connect(self._quit_from_tray)

        if self._tray_enabled:
            self._setup_tray_icon()

        tb = self.addToolBar("Main")
        # 指定順: NewWindow, Open, Open(NewWindow), Fit, PageUp, PageDown, PrevFile, NextFile
        tb.addAction(self.act_new_window)
        tb.addAction(self.act_open)
        tb.addAction(self.act_open_new)
        tb.addAction(self.act_fit)
        tb.addAction(self.act_prev)
        tb.addAction(self.act_next)
        tb.addAction(self.act_prev_file)
        tb.addAction(self.act_next_file)

        self._dir_files = []
        self._dir_file_index = -1

        self.statusBar().showMessage("Open / PgUp/PgDn=Prev/Next / Wheel=Zoom / Drag=Pan")
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
        menu.addAction(self.act_show_window)
        menu.addAction(self.act_hide_window)
        menu.addSeparator()
        menu.addAction(self.act_new_window)
        menu.addAction(self.act_open)
        menu.addAction(self.act_open_new)
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
        self._allow_close = True
        if self._tray_icon is not None:
            self._tray_icon.hide()
        app = QApplication.instance()
        if app is not None:
            app.quit()

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

        self.act_prev.setEnabled(has_file and pc > 1 and pi > 0)
        self.act_next.setEnabled(has_file and pc > 1 and (pc == 0 or pi < pc - 1))
        self.act_fit.setEnabled(self.view.has_image())
        self.act_next_file.setEnabled(bool(self._neighbor_file_path(1)))
        self.act_prev_file.setEnabled(bool(self._neighbor_file_path(-1)))
        if self._tray_available and self._tray_icon is not None:
            visible = self.isVisible()
            self.act_show_window.setEnabled(not visible)
            self.act_hide_window.setEnabled(visible)

        if has_file:
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
            self.statusBar().showMessage(f"Open / PgUp/PgDn=Prev/Next / Wheel=Zoom / Drag=Pan  Mode {mode_text}")

    def open_file(self):
        if not self.isVisible():
            self._show_main_window()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open TIFF",
            _default_open_directory(),
            "TIFF Files (*.tif *.tiff)",
        )
        if not path:
            log_debug("MainWindow open_file canceled")
            return

        log_info("MainWindow open_file selected path=%s", path)
        self._open_path(path)

    def open_file_in_new_window(self):
        if not self.isVisible():
            self._show_main_window()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open TIFF in New Window",
            _default_open_directory(),
            "TIFF Files (*.tif *.tiff)",
        )
        if not path:
            log_debug("MainWindow open_file_in_new_window canceled")
            return
        self.new_window_requested.emit(path)
        log_info("MainWindow requested new window path=%s", path)

    def _request_new_window(self):
        self.new_window_requested.emit("")
        log_info("MainWindow requested new empty window")

    @Slot(str)
    def _open_dropped_file(self, path: str):
        log_info("MainWindow drop_open path=%s", path)
        self._open_path(path)

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
            current_resolved = str(current.resolve(strict=False))
            files = [
                str(p.resolve(strict=False))
                for p in current.parent.iterdir()
                if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
            ]
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
        super().closeEvent(event)


class AppController(QObject):
    def __init__(self, app):
        super().__init__(app)
        self._app = app
        self._windows = []
        self._primary_window = None
        self._instance_server = SingleInstanceServer(self._on_ipc_message, self)

    def start_single_instance_server(self):
        self._instance_server.start()

    def startup(self, cli_args):
        self._primary_window = self._create_window(enable_tray=True)
        self._primary_window.resize(DEFAULT_WINDOW_SIZE[0], DEFAULT_WINDOW_SIZE[1])
        if cli_args:
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
        w.destroyed.connect(lambda _=None, win=w: self._on_window_destroyed(win))
        self._windows.append(w)
        log_info("AppController window created enable_tray=%s total=%s", enable_tray, len(self._windows))
        return w

    def _on_window_destroyed(self, window):
        for i, w in enumerate(list(self._windows)):
            if w is window:
                self._windows.pop(i)
                break
        log_info("AppController window destroyed total=%s", len(self._windows))

    @Slot(str)
    def open_new_window(self, path: str = ""):
        normalized_path = _normalize_input_path(path) if path else ""
        w = self._create_window(enable_tray=False)
        w.resize(DEFAULT_WINDOW_SIZE[0], DEFAULT_WINDOW_SIZE[1])
        w.show()
        log_info("AppController open_new_window path=%s", normalized_path if normalized_path else "(none)")
        if normalized_path:
            QTimer.singleShot(0, lambda p=normalized_path, win=w: win.open_from_cli_args([p]))

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
