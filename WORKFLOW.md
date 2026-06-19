# Workflow Proyek Anomaly NIDS

Dok ini menjelaskan workflow proyek dari sisi arsitektur, alur runtime, file, dan fungsi. Fokus utama: live capture jaringan, ekstraksi fitur, inferensi dual-stage LSTM-AE, penyimpanan PostgreSQL, dashboard, dan update model.

## Ringkasan Arsitektur

```text
Network interface
  -> CICFlowMeter
  -> completed-flow CSV
  -> traffic-source/extract_feature.py
  -> feature_schema.py ordering
  -> lstm-ae/model.py LSTMAEService
  -> database.py PostgreSQL
  -> dashboard/app.py Flask API/UI
```

Komponen utama:

- `main.py`: orkestrator live pipeline.
- `traffic-source/extract_feature.py`: parser output CICFlowMeter CSV dan pembuat 19 fitur runtime.
- `feature_schema.py`: daftar variant model, mapping nama fitur model ke key internal, ordering fitur.
- `lstm-ae/model.py`: arsitektur dual-stage LSTM autoencoder dan service runtime.
- `database.py`: schema DB, insert flow, query dashboard, query training/update model.
- `dashboard/app.py`: Flask dashboard dan API.
- `update_model/update_model.py`: background worker retrain incremental dan threshold recalibration.
- `state.py`: state global untuk dashboard dan pipeline.

## Runtime Workflow

1. User menjalankan `main.py`.
2. `main.py` baca env seperti `MODEL_VARIANT`, `DB_*`, `CAPTURE_INTERFACE`, `CICFLOWMETER_CMD`, `ENABLE_UPDATE_MODEL`, dan `ENABLE_LEARNING_MODE`.
3. `database.init_db()` membuat tabel `flows` dan `alerts`, lalu menambah 19 kolom fitur jika belum ada.
4. `LSTMAEService` load artifact dari `lstm-ae/dual-stage-ae/artifacts/<variant>/`.
5. Dashboard Flask start di thread terpisah.
6. Capture start di interface, default `eth0`, bisa `wlan0` lewat `CAPTURE_INTERFACE=wlan0`.
7. `start_cicflowmeter_capture()` menjalankan `.venv/bin/cicflowmeter` atau override dari `CICFLOWMETER_CMD`.
8. CICFlowMeter menulis CSV completed-flow ke `CICFLOWMETER_OUTPUT_FILE`.
9. `read_new_cicflowmeter_rows()` membaca row baru dari CSV.
10. `extract_features_from_row()` map kolom CICFlowMeter ke 19 fitur runtime.
11. Setiap row completed-flow masuk `get_feature_vector()`, scoring LSTM-AE, dan `database.insert_flow()`.
12. `LSTMAEService.add_to_buffer()` membentuk sliding window sepanjang `seq_len`.
13. Jika buffer sudah penuh, `LSTMAEService.predict()` menghitung error dual-stage dan anomaly score.
14. `LSTMAEService.is_anomaly()` membandingkan error dengan threshold.
15. Jika learning mode aktif, anomaly dipaksa `False`.
16. Dashboard API membaca completed flow DB untuk traffic chart, alert list, detail alert, dan kontribusi fitur.

## Model Workflow

Artifact aktif:

```text
lstm-ae/dual-stage-ae/artifacts/<variant>/metadata.json
lstm-ae/dual-stage-ae/artifacts/<variant>/model.pt
lstm-ae/dual-stage-ae/artifacts/<variant>/scaler.pkl
```

Variant didukung:

- `mrmr`
- `mutual_information`
- `rf_importance`
- `rfe`

Alur scoring:

1. Input sequence shape: `(time_steps, selected_feature_count)`.
2. Sequence diskalakan dengan `StandardScaler`.
3. Stage 1 merekonstruksi input.
4. Residual dihitung: `abs(input - recon1)`.
5. Stage 2 merekonstruksi residual.
6. Combined error dihitung dari error stage 1 + error stage 2.
7. `anomaly_score` diskalakan ke range maksimal 100.
8. Anomaly jika `reconstruction_error > threshold`.

## Update Model Workflow

Jika `ENABLE_UPDATE_MODEL=1`:

1. `update_model_worker()` menunggu `state.lstm_service` siap.
2. Worker tidur selama `UPDATE_MODEL_INTERVAL_SECONDS`.
3. `update_model_once()` mengambil normal flows dari DB dengan feature set `dual_stage_v1`.
4. Feature row diurutkan sesuai `selected_features` model aktif.
5. Jika `UPDATE_MODEL_ADAPT_SCALER=1`, scaler diadaptasi dengan `partial_fit`.
6. Data dibuat menjadi sequence batch.
7. Stage 1 dilatih untuk rekonstruksi input.
8. Stage 1 dipakai untuk menghasilkan residual.
9. Stage 2 dilatih untuk rekonstruksi residual.
10. Threshold recalibration memakai percentile combined error.
11. Model, scaler, dan metadata disimpan kembali ke artifact variant aktif.

## Database Workflow

Tabel utama:

- `flows`: completed flow level, metadata jaringan, 19 fitur runtime, score, status anomaly.
- `alerts`: subset flow yang anomaly, dipakai dashboard alert.

Versi fitur:

- Row baru memakai `feature_set_version = "dual_stage_v1"`.
- Query sequence model mengabaikan row lama yang bukan `dual_stage_v1`.

## Dashboard Workflow

Page:

- `/`: traffic dashboard.
- `/alerts`: list alert.
- `/alerts/<id>`: detail alert.

API:

- `/api/traffic-stats`: chart traffic.
- `/api/recent-flows`: recent flow table.
- `/api/top-talkers`: top source IP.
- `/api/protocol-distribution`: distribusi protocol.
- `/api/top-ports`: top destination port.
- `/api/alerts`: alert list.
- `/api/alert-summary`: ringkasan alert.
- `/api/alerts/<id>/detail`: detail alert dan contribution fitur.
- `/api/learning_mode`: aktifkan learning mode.
- `/api/alerts/<id>/resolve`: resolve alert.
- `/api/alerts/<id>/confirm`: confirm alert, saat ini status disimpan sebagai resolved.
- `/api/alerts/<id>/false-positive`: tandai false positive.
- `/api/alerts/bulk`: update status alert banyak.
- `/api/alerts/auto-dismiss`: auto-dismiss alert berdasar score.

## File dan Fungsi

### `main.py`

Peran file: entrypoint live service. Menghubungkan DB, model, dashboard, CICFlowMeter capture, feature extraction, inference, dan insert DB.

- `CaptureProcess.__init__(proc)`: simpan handle proses CICFlowMeter.
- `CaptureProcess.poll()`: cek apakah proses CICFlowMeter sudah exit.
- `CaptureProcess.stop()`: terminate lalu kill fallback agar tidak ada orphan process.
- `CaptureProcess.stderr_text()`: baca stderr CICFlowMeter untuk logging error capture.
- `signal_handler(sig, frame)`: set flag `running=False` saat SIGINT/SIGTERM.
- `start_cicflowmeter_capture(interface="eth0", output_file=...)`: validasi interface, start `.venv/bin/cicflowmeter` atau command override `CICFLOWMETER_CMD`.
- `run_pipeline()`: alur utama live capture sampai insert DB dan alert.
- `start_dashboard()`: start Flask dashboard di `0.0.0.0:5000`.

### `feature_schema.py`

Peran file: kontrak fitur model. Semua variant pakai nama fitur model, lalu dipetakan ke key internal DB/extractor.

- `normalize_feature_names(feature_names)`: ubah input fitur menjadi nama model canonical. Menerima nama model seperti `Dst Port` atau key internal seperti `dst_port_feat`.
- `feature_keys_for_names(feature_names)`: ubah daftar nama fitur model menjadi daftar key internal.
- `ordered_feature_values(features_dict, selected_features=None)`: ambil value fitur dari dict extractor sesuai urutan metadata model.
- `feature_row_to_vector(row, selected_features=None)`: ambil value fitur dari row DB sesuai urutan metadata model.
- `_self_check()`: assert kecil untuk cek union 19 fitur dan backward compatibility key lama.

### `traffic-source/extract_feature.py`

Peran file: parse CSV CICFlowMeter, map header ke 19 fitur runtime, dan output metadata flow completed.

- `safe_float(val, default=0.0)`: parse float aman dari field CICFlowMeter kosong, invalid, `NaN`, atau `Infinity`.
- `safe_int(val, default=0)`: parse int aman dari field CICFlowMeter kosong atau invalid.
- `normalize_column(name)`: normalisasi nama kolom agar beda spasi/slash/underscore tetap bisa dimatch.
- `normalized_row(row)`: buat dict row dengan key kolom ternormalisasi.
- `first_value(row, aliases, default="")`: ambil nilai pertama dari daftar alias header.
- `parse_timestamp(value)`: ubah timestamp CICFlowMeter numeric/string menjadi epoch seconds.
- `parse_protocol(value)`: normalisasi protocol angka/string menjadi `TCP`, `UDP`, `ICMP`, atau uppercase.
- `extract_features_from_row(row)`: map satu row CICFlowMeter menjadi `{features, metadata}`.
- `read_new_cicflowmeter_rows(path, offset=0, header=None)`: tail CSV CICFlowMeter dan return row baru tanpa duplikasi.
- `process_csv_stream(input_stream, callback=None)`: proses file/stream CSV CICFlowMeter.
- `get_feature_vector(features_dict, selected_features=None)`: output vector fitur sesuai urutan model aktif.

### `lstm-ae/model.py`

Peran file: model PyTorch dan service runtime.

- `LSTMAutoencoder.__init__(num_features, time_steps, hidden_dim=64, dropout=0.2)`: buat encoder LSTM, decoder LSTM, dan output layer.
- `LSTMAutoencoder.forward(x)`: rekonstruksi sequence input.
- `DualStageAutoencoder.__init__(num_features, time_steps, hidden_dim=64, dropout=0.2)`: buat stage 1 dan stage 2 autoencoder.
- `DualStageAutoencoder.forward(x)`: output `recon1` dan rekonstruksi residual.
- `dual_stage_errors(model, x)`: hitung combined error per sequence dan per feature.
- `LSTMAEService.__init__(config)`: load config, metadata, model, scaler, threshold, dan buffer runtime.
- `LSTMAEService._load_metadata()`: baca `metadata.json`.
- `LSTMAEService._load_model()`: load `model.pt` ke `DualStageAutoencoder`.
- `LSTMAEService._load_scaler()`: load `scaler.pkl` dan validasi sudah fitted.
- `LSTMAEService.save_model()`: simpan state dict model.
- `LSTMAEService.save_scaler()`: simpan scaler.
- `LSTMAEService.save_metadata()`: simpan metadata runtime terbaru, termasuk threshold.
- `LSTMAEService.adapt_scaler(features)`: update scaler dengan `partial_fit`.
- `LSTMAEService.update_threshold(threshold, threshold_percentile=None)`: update threshold dan persist metadata/model.
- `LSTMAEService.recalibrate_threshold(data, threshold_percentile=None)`: hitung percentile error baru dan update threshold.
- `LSTMAEService.scale_features(features)`: transform fitur 1D/2D dengan scaler.
- `LSTMAEService.scale_sequence(sequence)`: scale satu sequence 2D.
- `LSTMAEService.scale_sequence_batch(data)`: scale batch sequence 3D.
- `LSTMAEService.add_to_buffer(feature_vector)`: append fitur live ke sliding buffer; return scaled sequence jika buffer penuh.
- `LSTMAEService.get_feature_names()`: return selected features aktif.
- `LSTMAEService.predict(sequence)`: hitung reconstruction error, anomaly score, dan feature error.
- `LSTMAEService.is_anomaly(reconstruction_error)`: boolean anomaly berdasar threshold.
- `LSTMAEService.train_incremental(data, epochs=1, lr=1e-4)`: retrain stage 1 dan stage 2 secara incremental.
- `LSTMAEService.compute_reconstruction_errors(data)`: hitung error batch untuk recalibration.
- `LSTMAEService.get_model()`: return model PyTorch.
- `LSTMAEService.get_scaler()`: return scaler.
- `_self_check()`: smoke check shape model dual-stage.

### `database.py`

Peran file: akses PostgreSQL dan query semua data runtime/dashboard/update model.

- `get_connection()`: buka koneksi PostgreSQL dari env `DB_*`.
- `init_db()`: buat tabel `flows` dan `alerts`, tambah kolom fitur idempotent.
- `insert_flow(metadata, features, anomaly_score, is_anomaly)`: insert row flow dan insert alert jika anomaly.
- `get_recent_flows(limit=100)`: ambil flow terbaru untuk dashboard.
- `get_traffic_stats(minutes=30)`: agregasi traffic per minute/hour/day.
- `get_alerts(limit=0, status=None, minutes=None)`: query alert list dengan filter status/waktu.
- `get_top_talkers(minutes=30, limit=10)`: top source IP berdasar jumlah flow.
- `get_protocol_distribution(minutes=30)`: agregasi flow per protocol.
- `get_top_ports(minutes=30, limit=10)`: top destination port.
- `get_alert_summary(minutes=None)`: ringkasan total/open/resolved/false positive/avg score.
- `resolve_alert(alert_id)`: ubah status alert menjadi `Resolved`.
- `get_alert_detail(alert_id)`: ambil satu alert untuk detail page.
- `false_positive_alert(alert_id)`: ubah status alert menjadi `False Positive`.
- `confirm_alert(alert_id)`: saat ini sama seperti resolve, menyimpan status `Resolved`.
- `bulk_update_alerts(alert_ids, status)`: update status beberapa alert sekaligus.
- `bulk_resolve_by_score(max_score)`: tandai alert aktif dengan score di bawah limit sebagai false positive.
- `get_normal_flows(limit=1000, feature_set_version=FEATURE_SET_VERSION, selected_features=None)`: ambil flow normal untuk update model.
- `get_flow_feature_sequence(flow_id, limit, feature_set_version=FEATURE_SET_VERSION, selected_features=None)`: ambil sequence fitur sebelum/sampai flow tertentu untuk kontribusi alert.

### `dashboard/app.py`

Peran file: Flask web app dan API JSON untuk dashboard.

- `normalize_alert_status(status)`: tampilkan `Active` sebagai `Open`, `Confirmed` sebagai `Resolved`.
- `add_no_cache(response)`: tambah header no-cache untuk endpoint `/api/*`.
- `_json_payload()`: baca JSON POST aman, body kosong menjadi `{}`.
- `_coerce_non_negative_int(value, default, field_name)`: validasi/cast int non-negatif.
- `_coerce_float(value, default, field_name)`: validasi/cast float.
- `_json_error(message, status_code=400)`: return JSON error standar.
- `index()`: render `traffic.html`.
- `alerts_page()`: render `alerts.html`.
- `alert_detail_page(alert_id)`: render `alert_detail.html`.
- `api_traffic_stats()`: return traffic stats untuk chart.
- `api_recent_flows()`: return recent flows.
- `api_top_talkers()`: return top source IP.
- `api_protocol_dist()`: return protocol distribution.
- `api_top_ports()`: return top destination ports.
- `api_alerts()`: return alert list.
- `api_alert_summary()`: return summary alert.
- `api_resolve_alert(alert_id)`: resolve alert.
- `api_learning_mode()`: aktifkan learning mode durasi tertentu.
- `api_alert_detail(alert_id)`: return detail alert dan feature contribution dari model aktif.
- `api_confirm_alert(alert_id)`: confirm alert; saat ini disimpan sebagai resolved.
- `api_false_positive_alert(alert_id)`: tandai alert sebagai false positive.
- `api_bulk_alerts()`: update banyak alert dengan validasi payload.
- `api_auto_dismiss()`: false-positive-kan alert aktif dengan score di bawah limit.

### `state.py`

Peran file: state global kecil yang dipakai pipeline dan dashboard.

- `AppState.__init__()`: set default learning mode off dan `lstm_service=None`.
- `AppState.enable_learning_mode(duration_minutes)`: aktifkan learning mode sampai waktu tertentu.
- `AppState.disable_learning_mode()`: matikan learning mode.
- `state`: singleton global `AppState`.

### `update_model/update_model.py`

Peran file: retraining incremental dan threshold recalibration.

- `build_sequences(flow_vectors, seq_len)`: ubah matrix flow menjadi sliding sequence batch.
- `update_model_once(lstm_service, config=None)`: ambil normal flows, adapt scaler, train model, recalibrate threshold, return summary.
- `update_model_worker(config=None)`: loop background periodik untuk memanggil `update_model_once()`.

### `smoke_dual_stage.py`

Peran file: smoke test ringan tanpa test framework untuk model dan artifact.

- `config_for_variant(variant)`: buat config service untuk artifact variant tertentu.
- `check_dual_stage_shapes()`: cek shape forward model dan error dual-stage.
- `check_variant_metadata()`: cek metadata semua variant cocok dengan schema.
- `check_service_smoke()`: load tiap variant, scale dummy sequence, dan predict.
- `main()`: jalankan semua smoke check.

### `smoke_dashboard_routes.py`

Peran file: smoke test ringan untuk API dashboard tanpa DB live.

- `main()`: monkeypatch fungsi DB yang diperlukan, lalu cek route POST penting dan validasi payload.

### `dashboard/templates/traffic.html`

Peran file: UI dashboard traffic. Memanggil API traffic, recent flows, top talkers, protocol distribution, dan top ports.

### `dashboard/templates/alerts.html`

Peran file: UI daftar alert. Memanggil API alert list/summary, filter status/waktu, bulk action, auto-dismiss.

### `dashboard/templates/alert_detail.html`

Peran file: UI detail alert. Memanggil API detail alert, menampilkan metadata flow dan kontribusi fitur.

## Artifact dan Direktori Non-runtime

### `lstm-ae/dual-stage-ae/artifacts/`

Isi artifact model aktif per variant:

- `mrmr/`
- `mutual_information/`
- `rf_importance/`
- `rfe/`

Tiap folder berisi:

- `metadata.json`: selected features, time steps, hidden dim, threshold.
- `model.pt`: state dict PyTorch.
- `scaler.pkl`: scaler fitted.

### `lstm-ae/dual-stage-ae/architecture/`

Notebook eksperimen/training dual-stage. Bukan bagian runtime service.

### `lstm-ae/new_lstm_model/`

Artifact/model lama. Runtime baru memakai `dual-stage-ae/artifacts/<variant>`.

## Env Penting

- `MODEL_VARIANT`: `mrmr`, `mutual_information`, `rf_importance`, atau `rfe`. Default `mrmr`.
- `MODEL_DEVICE`: device PyTorch, default `cpu`.
- `CAPTURE_INTERFACE`: interface capture, contoh `wlan0`.
- `CICFLOWMETER_CMD`: optional override command CICFlowMeter. Placeholder tersedia: `{interface}`, `{output_file}`.
- `CICFLOWMETER_OUTPUT_FILE`: file CSV output CICFlowMeter, default `/tmp/anomaly-nids-cicflowmeter/flows.csv`.
- `CICFLOWMETER_POLL_SECONDS`: interval polling CSV, default `1`.
- `ENABLE_UPDATE_MODEL`: `1` untuk worker retrain, `0` untuk off.
- `UPDATE_MODEL_INTERVAL_SECONDS`: interval worker.
- `UPDATE_MODEL_MIN_NORMAL_FLOWS`: minimum normal flow sebelum retrain.
- `UPDATE_MODEL_BATCH_LIMIT`: batas flow training.
- `UPDATE_MODEL_EPOCHS`: epoch train incremental.
- `UPDATE_MODEL_LEARNING_RATE`: learning rate.
- `UPDATE_MODEL_ADAPT_SCALER`: `1` untuk scaler `partial_fit`.
- `UPDATE_MODEL_THRESHOLD_PERCENTILE`: percentile threshold baru.
- `ENABLE_LEARNING_MODE`: `1` untuk learning mode saat startup.
- `LEARNING_MODE_DURATION_MINUTES`: durasi learning mode startup.
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASS`: koneksi PostgreSQL.

## Command Operasional

Run live service:

```bash
CAPTURE_INTERFACE=wlan0 .venv/bin/python main.py
```

Run tanpa update worker:

```bash
CAPTURE_INTERFACE=wlan0 ENABLE_UPDATE_MODEL=0 .venv/bin/python main.py
```

Run dengan PostgreSQL test container di port 55432:

```bash
DB_PORT=55432 CAPTURE_INTERFACE=wlan0 .venv/bin/python main.py
```

Smoke model:

```bash
.venv/bin/python smoke_dual_stage.py
```

Smoke dashboard routes:

```bash
.venv/bin/python smoke_dashboard_routes.py
```

One-shot update model:

```bash
.venv/bin/python update_model/update_model.py
```

## Catatan Perubahan Penting

- Runtime sudah dual-stage LSTM-AE.
- Capture runtime memakai CICFlowMeter external command, bukan Argus/ra.
- Runtime menyimpan 19 fitur union, tetapi model hanya membaca `selected_features` aktif.
- `feature_set_version` row baru adalah `dual_stage_v1`.
- Capture di `main.py` menjalankan CICFlowMeter sebagai child process agar shutdown bersih.
- Dashboard API menerima body POST kosong untuk route yang punya default, dan validasi payload sebelum menyentuh state/DB.
