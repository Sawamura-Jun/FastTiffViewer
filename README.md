<img width="1174" height="950" alt="snapshot_20260214_201846_bc" src="https://github.com/user-attachments/assets/082b2fa5-67b1-4012-b215-2c2729502b7b" />
<img width="1174" height="950" alt="snapshot_20260214_201911_bc" src="https://github.com/user-attachments/assets/0dd69755-c8fb-4178-9a94-fe52ca2ac5e2" />

# TIFF 高速ビューワー / Fast TIFF Viewer

---

## README（日本語）

### 概要

TIFF を **高速に表示**するためのシンプルなビューワーアプリです。マウス操作とショートカットで快適に閲覧できます。**マルチページ TIFF** に対応しています。

## 特長

- **高速表示**
- マウスホイールで **画像の拡大縮小**
- **Ctrl + ホイール**で **ウィンドウ自体の拡大縮小**
- **マルチページ TIFF** 対応

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

### 3) 初回起動時の .NET Runtime インストール（追加）

初回に `FastTiffViewerLauncher.exe` を起動した際、環境によっては次のように **.NET Runtime のインストール**を求められることがあります。  
その場合は、表示された案内に従って **インストールを実行**してください。

<img width="810" height="315" alt="snapshot_20260218_222732_bc" src="https://github.com/user-attachments/assets/dde0a2b1-a62c-4b26-a021-2ab9eb5c056b" />

## 使い方

### ファイルを開く

- メニュー **`Open`**：ファイルを開きます  
- **ドラッグ＆ドロップ（DnD）**：ウィンドウにファイルをドロップして開けます

### 表示操作

- マウスホイール：画像の拡大 / 縮小
- **Ctrl + マウスホイール**：ウィンドウの拡大 / 縮小

### フィット表示

- メニュー **`Fit(F)`**：ウィンドウサイズに画像をジャストフィット
- ショートカット：**`f`**

### ページ移動（マルチページ TIFF）

- 前ページ：**`PageUp`**（ショートカット：`PageUp`）
- 次ページ：**`PageDown`**（ショートカット：`PageDown`）

### 同一フォルダ内のファイル移動（名前順）

- メニュー **`PrevFile(B)`**：同じフォルダ内の **前のファイル（名前順）** を開く（ショートカット：**`b`**）
- メニュー **`NextFile(N)`**：同じフォルダ内の **次のファイル（名前順）** を開く（ショートカット：**`n`**）

---

## README（English）

### Overview

A simple viewer app designed to display TIFF images **quickly**. It provides smooth mouse controls, handy shortcuts, and supports **multi-page TIFF** files.

## Features

- **High-speed rendering**
- Mouse wheel to **zoom in/out the image**
- **Ctrl + Mouse Wheel** to **zoom the application window itself**
- **Multi-page TIFF** support

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

### 3) .NET Runtime installation on first launch (Added)

On the first launch of `FastTiffViewerLauncher.exe`, Windows may prompt you to install a **.NET Runtime** (depends on your environment).  
If that happens, please **run the installation** as instructed by the prompt.

<img width="810" height="315" alt="snapshot_20260218_222732_bc" src="https://github.com/user-attachments/assets/dde0a2b1-a62c-4b26-a021-2ab9eb5c056b" />

## Usage

### Open a file

- Menu **`Open`**: Open a file  
- **Drag & Drop (DnD)**: Drop a file onto the window to open it

### View controls

- Mouse Wheel: Zoom the image in/out
- **Ctrl + Mouse Wheel**: Zoom the application window in/out

### Fit to window

- Menu **`Fit(F)`**: Fit the image exactly to the window size
- Shortcut: **`f`**

### Page navigation (Multi-page TIFF)

- Previous page: **`PageUp`** (shortcut: `PageUp`)
- Next page: **`PageDown`** (shortcut: `PageDown`)

### Navigate files in the same folder (name order)

- Menu **`PrevFile(B)`**: Open the **previous file (by name order)** in the same folder (shortcut: **`b`**)
- Menu **`NextFile(N)`**: Open the **next file (by name order)** in the same folder (shortcut: **`n`**)
