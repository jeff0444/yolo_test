# YOLOv8 Object Detector with FFmpeg Recording

這是一個基於 Ultralytics YOLOv8 的物件辨識程式，支援即時串流 (RTSP)、影片檔、單張圖片與資料夾圖片批次處理。程式具備彈性的錄影功能，可透過 FFmpeg (PyAV) 將辨識結果儲存為 H.264 編碼的 MP4 影片，並支援條件式錄影（只在偵測到物件時儲存）。

## 主要功能

*   **多源輸入支援**：支援 RTSP 串流、影片檔 (MP4, AVI 等)、單張圖片、資料夾批次處理。
*   **高效能串流讀取**：針對 RTSP 串流採用多執行緒緩衝讀取，確保即時性。
*   **FFmpeg 錄影輸出**：
    *   使用 `pyav` 進行 H.264 編碼，相容性高。
    *   自動 FPS 偵測與校正 (針對 RTSP 串流自動估算 FPS)。
    *   可調整 CRF (Constant Rate Factor) 參數控制壓縮品質。
*   **彈性儲存模式**：
    *   `all`：儲存完整辨識過程影片。
    *   `key`：僅儲存偵測到物件的前後 3 秒片段 (Smart Clipping)。
*   **視覺化進度條**：在處理既有影片且選擇完整儲存時，會在影片下方繪製時間軸，橘色代表處理進度，紅色標記偵測到物件的時間點。
*   **即時顯示**：可選參數開啟視窗即時監看辨識結果。

## 環境需求

請確保安裝以下 Python 套件：

```bash
pip install ultralytics opencv-python numpy av tqdm torch
```
*注意：需安裝 [PyAV](https://github.com/PyAV-Org/PyAV) (`av`) 以支援影片編碼功能。*

## 使用說明

### 基本指令

```bash
python detector.py --source <輸入來源> [參數]
```

### 參數列表

| 參數 | 說明 | 預設值 |
| :--- | :--- | :--- |
| `--source` | **[必填]** 輸入來源路徑 (檔案、資料夾或 URL) | 無 |
| `--model` | YOLO 模型路徑 | `yolov8n.pt` |
| `--conf` | 信心度閥值 (Confidence Threshold) | `0.45` |
| `--iou` | IOU 閥值 (NMS) | `0.45` |
| `--device` | 運算裝置 (`cpu`, `cuda`, `0` 等) | 自動偵測 |
| `--output` | 輸出根目錄 | `output` |
| `--show` | 是否開啟視窗即時顯示結果 | `False` |
| `--save_mode` | 儲存模式：`all` (全部) 或 `key` (重點片段) | `all` |

### 執行範例

**1. 辨識影片檔並完整儲存 (預設)**
```bash
python detector.py --source input/video.mp4
```
輸出結果會依模型名稱分類，例如 `output/yolov8n/out_video.mp4`，影片下方會附加偵測時間軸。

**2. 辨識 RTSP 串流並顯示畫面**
```bash
python detector.py --source rtsp://admin:pass@192.168.1.10:554/stream --show
```
串流結果會以時間戳記命名儲存。

**3. 僅儲存偵測到物件的片段 (重點模式)**
```bash
python detector.py --source input/cctv_record.mp4 --save_mode key
```
此模式會啟用緩衝機制，只將偵測到物件的前 3 秒至後 3 秒的片段寫入檔案，大幅節省儲存空間。

**4. 批次處理圖片資料夾**
```bash
python detector.py --source input/images_folder --conf 0.5
```
會在 `output` 下建立對應的資料夾並儲存辨識後的圖片。

**5. 指定模型與 GPU 裝置**
```bash
python detector.py --source video.mp4 --model models/yolov8x.pt --device 0
```

## 輸出結構

程式會根據使用的模型名稱自動建立子資料夾，避免混淆不同模型的測試結果。

```
output/
├── yolov8n/
│   ├── rtsp_20250112_103000.mp4
│   ├── out_test_video.mp4
│   └── test_folder/
│       ├── img1.jpg
│       └── img2.jpg
└── custom_model/
    └── out_video.mp4
```

## 進階設定

若需調整影片壓縮品質，可修改 `saveStream_ffmpeg.py` 中的 `crf` 參數 (預設為 35)：

```python
# saveStream_ffmpeg.py
class SaveStream_ffmpeg:
    def __init__(self, filename, fps=-1, width=640, height=480, crf=35):
        # 值越小畫質越高但檔案越大 (建議範圍 18-28，監控用途可設 30-35)
```
