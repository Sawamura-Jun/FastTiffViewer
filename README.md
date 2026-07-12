# TIFF 高速ビューワー / Fast TIFF Viewer

<p align="center">
  <img width="600" alt="Fast TIFF Viewer のメイン画面" src="https://github.com/user-attachments/assets/082b2fa5-67b1-4012-b215-2c2729502b7b" />
  <br />
  <img width="600" alt="Fast TIFF Viewer のフィット表示" src="https://github.com/user-attachments/assets/05ebb83c-d02a-442e-8fbb-2f12822598e9" />
  <br />
  <img width="600" alt="Fast TIFF Viewer の画像表示" src="https://github.com/user-attachments/assets/8e32601d-326c-41e9-bbaa-236f8a1db39c" />
  <br />
  <img width="600" alt="Fast TIFF Viewer の見開き表示" src="https://github.com/user-attachments/assets/e38f1835-2c57-473e-9898-ade198719153" />
</p>

---

## README（日本語）

### 概要

TIFF を **高速に表示**するためのシンプルなビューワーアプリです。マウス操作とショートカットで快適に閲覧できます。**マルチページ TIFF** に対応しています。

## 特長

- **高速表示**
- マウスホイールで **画像の拡大縮小**
- **Ctrl + ホイール**で **ウィンドウ自体の拡大縮小**
- **マルチページ TIFF** 対応
- **トリミング保存** 対応
- 表示中の TIFF を **デスクトップへ保存**
- 2つの TIFF の変更箇所を色分けした **差分 TIFF を生成**
- **`c`キー**でアクティブなウィンドウを閉じる
- **Spread(S)** で2つのウィンドウを左右に見開き配置し、表示操作を同期
- **Fit(F)** で画像をウィンドウにフィット
- **ダブルクリック**でウィンドウの最大化 / 通常表示を切り替え
- 画像の縦横比に合わせて起動時のウィンドウサイズを自動調整

## 実行ファイルと起動方法（追加機能）

### 1) `FastTiffViewer.exe`（本体）

- **引数なし**で実行すると、アプリは **システムトレイに常駐**します。  
- **TIFF ファイルのパスを引数に指定**して実行すると、起動して **指定した画像を表示**します。  
  - 例：右クリック → **「プログラムから開く」** で TIFF を開く場合

### 2) `FastTiffViewerLauncher.exe`（起動高速化ランチャー / 追加）

- **2回目以降の起動速度を高める**ためのランチャーです。  
- `FastTiffViewer.exe` がすでに **システムトレイに常駐**している場合でも、`FastTiffViewerLauncher.exe` を経由することで **さらに起動（表示）を高速化**できるようにしています（常駐プロセスへの受け渡しを最適化）。  
- Windows の **ファイル関連付け**（右クリック →「プログラムから開く」/ 既定のアプリ）を **`FastTiffViewerLauncher.exe`** に設定すると効果的です。  
- ランチャーが起動すると、**本体の `FastTiffViewer.exe` を起動**し、必要に応じて **システムトレイに常駐**させます。  
  - 以後、TIFF を開く操作を繰り返したときの体感起動が速くなります。

## 使い方

### ファイルを開く

- メニュー **`Open`**：ファイルを開きます  
- **ドラッグ＆ドロップ（DnD）**：ウィンドウにファイルをドロップして開けます

### 表示操作

- マウスホイール：画像の拡大 / 縮小
- **Ctrl + マウスホイール**：ウィンドウの拡大 / 縮小
- 画像表示領域を左ダブルクリック：ウィンドウの最大化 / 通常表示を切り替え

### ウィンドウ操作

- メニュー **`New`** / ショートカット：**`Ctrl + Shift + N`** で新しいウィンドウを開きます
- ショートカット：**`c`** でアクティブなウィンドウを閉じます（見開き同期中はペアの両方を閉じます）
- TIFF を指定して起動した場合、画像表示領域の長辺が 1000 px になるよう、画像の縦横比に合わせて初期ウィンドウサイズを調整します

### 見開き配置・同期

- メニュー **`Spread(S)`** / ショートカット：**`s`** で、同じモニター上の2つのウィンドウを左右に見開き配置して同期します
- 同期中は、パン、ズーム、フィット、ページ移動、同一フォルダ内のファイル移動、`c` キーによるクローズが両方のウィンドウに反映されます
- 同じペアで **`Spread(S)`** / **`s`** をもう一度実行すると同期を解除し、再度実行すると同期を有効にします

### 表示中の TIFF を保存

- メニュー **`Save`**：表示中の元 TIFF を同じファイル名でデスクトップへコピーします
- デスクトップに同名ファイルがある場合は、上書き確認を表示します

### トリミング

- 右ドラッグ：トリミング範囲を作成
- 範囲内クリック：トリミング画像を保存し、PNG としてクリップボードへコピー
- 範囲内ドラッグ：トリミング範囲を移動
- 4辺・4隅ドラッグ：トリミング範囲を調整
- 右クリック：トリミング範囲を非表示
- 保存先はツールバーの **`トリミング保存先`** に指定します

### フィット表示

- メニュー **`Fit(F)`** / ショートカット：**`f`** で画像をウィンドウサイズにジャストフィットします
- フィット表示中にもう一度実行すると、画像の表示倍率を保ったままウィンドウサイズを表示画像に合わせます

### ページ移動（マルチページ TIFF）

- 前ページ：**`PageUp`**（ショートカット：`PageUp`）
- 次ページ：**`PageDown`**（ショートカット：`PageDown`）

### 同一フォルダ内のファイル移動（名前順）

- メニュー **`PrevFile(B)`**：同じフォルダ内の **前のファイル（名前順）** を開く（ショートカット：**`b`**）
- メニュー **`NextFile(N)`**：同じフォルダ内の **次のファイル（名前順）** を開く（ショートカット：**`n`**）

### TIFF の差分検出

1. 比較元となる旧 TIFF と、比較先となる新 TIFF を別々のウィンドウで開きます。
2. **新 TIFF 側のウィンドウ**でメニュー **`Diff(D)`** またはショートカット **`d`** を実行します。
3. 新 TIFF に対する追加箇所は **赤**、削除箇所は **青**で表示したマルチページ差分 TIFF を生成します。完全一致したページには **「差分検出なし」**と表示します。
4. 処理中はステータスバーに進捗を表示し、完了後は生成した差分 TIFF を新しいウィンドウで開きます。

- 差分 TIFF の保存先はツールバーの **`Crop save as`** 欄で指定します
- 保存先を指定しない場合は、新 TIFF と同じフォルダへ **`<新TIFF名>_dif.tif`** または **`<新TIFF名>_dif.tiff`** として保存します
- ページ数やページごとの画像サイズが異なる TIFF も比較できます

---

## README（English）

### Overview

A simple viewer app designed to display TIFF images **quickly**. It provides smooth mouse controls, handy shortcuts, and supports **multi-page TIFF** files.

## Features

- **High-speed rendering**
- Mouse wheel to **zoom in/out the image**
- **Ctrl + Mouse Wheel** to **zoom the application window itself**
- **Multi-page TIFF** support
- **Crop save** support
- Save the displayed TIFF to the **Desktop**
- Generate a color-coded **difference TIFF** from two TIFF files
- **`c` key** to close the active window
- **Spread(S)** to arrange two windows side by side and synchronize their views
- **Fit(F)** to fit the image to the window
- **Double-click** to toggle window maximize / restore
- Automatically size the initial window for the image aspect ratio

## Executables & startup behavior (Added)

### 1) `FastTiffViewer.exe` (Main app)

- When launched **without arguments**, it stays **resident in the system tray**.  
- When launched **with a TIFF file path as an argument**, it starts and **opens the specified image**.  
  - Example: opening a TIFF via Windows **“Open with…”**

### 2) `FastTiffViewerLauncher.exe` (Startup accelerator / Added)

- A small launcher to **improve perceived startup speed** on the **2nd and later** opens.  
- Even when `FastTiffViewer.exe` is already running in the system tray, opening files via the launcher can be even faster (it optimizes the handoff to the resident process).  
- For best results, set Windows file association (“Open with…”, default app) for `.tif/.tiff` to **`FastTiffViewerLauncher.exe`**.  
- The launcher will start the main app (**`FastTiffViewer.exe`**) and keep it **resident in the system tray** when needed.  
  - Subsequent opens are typically faster because the main process is already available.

## Usage

### Open a file

- Menu **`Open`**: Open a file  
- **Drag & Drop (DnD)**: Drop a file onto the window to open it

### View controls

- Mouse Wheel: Zoom the image in/out
- **Ctrl + Mouse Wheel**: Zoom the application window in/out
- Left double-click in the image view: Toggle window maximize / restore

### Window controls

- Menu **`New`** / shortcut: **`Ctrl + Shift + N`** opens a new window
- Shortcut: **`c`** closes the active window (or both paired windows while spread synchronization is enabled)
- When launched with a TIFF file, the initial window follows the image aspect ratio and sets the long edge of the image view to 1000 px

### Spread layout and synchronization

- Menu **`Spread(S)`** / shortcut: **`s`** arranges and synchronizes two windows side by side on the same monitor
- While synchronized, pan, zoom, fit, page navigation, same-folder file navigation, and closing with `c` are applied to both windows
- Run **`Spread(S)`** / **`s`** again for the same pair to disable synchronization; run it once more to re-enable synchronization

### Save the displayed TIFF

- Menu **`Save`**: Copy the currently displayed source TIFF to the Desktop with the same file name
- If a file with the same name already exists on the Desktop, the app asks before overwriting it

### Crop

- Right drag: Create a crop area
- Click inside the area: Save the cropped image and copy it to the clipboard as PNG
- Drag inside the area: Move the crop area
- Drag edges/corners: Adjust the crop area
- Right click: Hide the crop area
- Set the output path in **`トリミング保存先`** on the toolbar

### Fit to window

- Menu **`Fit(F)`** / shortcut: **`f`** fits the image exactly to the window size
- Run it again while in Fit mode to resize the window to the displayed image without changing the image scale

### Page navigation (Multi-page TIFF)

- Previous page: **`PageUp`** (shortcut: `PageUp`)
- Next page: **`PageDown`** (shortcut: `PageDown`)

### Navigate files in the same folder (name order)

- Menu **`PrevFile(B)`**: Open the **previous file (by name order)** in the same folder (shortcut: **`b`**)
- Menu **`NextFile(N)`**: Open the **next file (by name order)** in the same folder (shortcut: **`n`**)

### TIFF difference detection

1. Open the old TIFF and new TIFF in separate windows.
2. In the **new TIFF window**, run menu **`Diff(D)`** or press **`d`**.
3. The app creates a multi-page difference TIFF: additions are shown in **red**, deletions in **blue**, and identical pages are marked **“差分検出なし”** (no differences detected).
4. Progress appears in the status bar, and the generated difference TIFF opens in a new window when processing completes.

- Set the difference TIFF destination in the toolbar's **`Crop save as`** field
- If no destination is specified, the result is saved beside the new TIFF as **`<new TIFF name>_dif.tif`** or **`<new TIFF name>_dif.tiff`**
- TIFF files with different page counts or per-page image sizes can also be compared
