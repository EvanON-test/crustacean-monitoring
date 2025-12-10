# Legacy Code Archive

This directory contains the original implementation of the Crustacean Monitoring System before the refactor to a proper Python package structure.

## Files

### Main Scripts

| File | Description | New Equivalent |
|------|-------------|----------------|
| `pipeline.py` | Original offline video processing script | `scripts/run_offline.py` |
| `realtime_pipeline.py` | Original real-time headless processing | `scripts/run_realtime.py` |
| `realtime_pipeline_demo.py` | Original real-time with display | `scripts/run_realtime.py --display` |
| `monitoring.py` | Original hardware monitoring script | `scripts/run_monitoring.py` |

### Utility Modules

| File | Description | New Equivalent |
|------|-------------|----------------|
| `binary_classifier_util.py` | Binary classifier functions | `crustacean/models/binary_classifier.py` |
| `frame_selector_util.py` | Frame selector functions | `crustacean/models/frame_selector.py` |
| `object_detector_util.py` | Object detector functions | `crustacean/models/object_detector.py` |
| `keypoint_detector_util.py` | Keypoint detector functions | `crustacean/models/keypoint_detector.py` |

## Why Keep These?

These files are preserved for:
1. **Reference** - Understanding the original implementation
2. **Comparison** - Benchmarking new vs old performance
3. **Fallback** - In case issues arise with the new implementation
4. **Documentation** - Historical record of the codebase evolution

## Usage (Deprecated)

These scripts can still be run if needed, but you'll need to copy them back to the root directory:

```bash
# Copy a script back to root
cp legacy_code/pipeline.py ./

# Run it (requires original dependencies)
sudo python3.9 pipeline.py
```

## Migration

For the new implementation, use:

```bash
# Instead of: python pipeline.py
python scripts/run_offline.py --video-dir ./processing/video

# Instead of: python realtime_pipeline.py
python scripts/run_realtime.py

# Instead of: python realtime_pipeline_demo.py
python scripts/run_realtime.py --display

# Instead of: python monitoring.py
python scripts/run_monitoring.py --video-dir ./processing/video
```

See `docs/MIGRATION.md` for detailed migration instructions.

## Note

These files are **not maintained** and may not work with the current package structure. They are kept for reference only.
