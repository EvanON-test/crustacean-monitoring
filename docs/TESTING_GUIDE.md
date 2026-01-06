# Jetson Nano Testing Guide

A comprehensive, step-by-step guide for manually testing the refactored Crustacean Monitoring System on Jetson Nano hardware. This guide covers all aspects of Task 21 from the implementation plan.

> **Note:** This guide covers both Docker and native installation testing. Docker is the recommended approach for production deployments.

---

## Table of Contents

1. [Prerequisites & Setup](#prerequisites--setup)
2. [Docker Testing (Recommended)](#docker-testing-recommended)
3. [Task 21.1: Run Full Test Suite](#task-211-run-full-test-suite)
4. [Task 21.2: Hardware Testing](#task-212-test-on-jetson-nano-hardware)
5. [Task 21.3: Performance Benchmarking](#task-213-performance-benchmarking)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Results Template](#results-template)

---

## Prerequisites & Setup

### Hardware Requirements

Before starting testing, verify you have:

| Component | Requirement | How to Check |
|-----------|-------------|--------------|
| Jetson Nano | 2GB or 4GB model | `cat /proc/device-tree/model` |
| JetPack | 4.6.x recommended | `cat /etc/nv_tegra_release` |
| Camera | CSI or USB camera | `ls /dev/video*` |
| Storage | At least 8GB free | `df -h` |
| RAM | Monitor available | `free -h` |

### Network Setup (for remote testing)

If testing remotely via SSH:

```bash
# Find your Jetson's IP address
hostname -I

# From your development machine, test SSH connection
ssh jetson@<jetson-ip>

# For display tests over SSH, you'll need X11 forwarding
ssh -X jetson@<jetson-ip>
```

> **Note:** Display mode tests work best with a physical monitor connected to the Jetson. X11 forwarding over SSH can be slow and may affect performance measurements.

### Initial System Check

Run these commands on your Jetson to verify the system is ready:

```bash
# Check system info
uname -a
# Expected: Linux jetson 4.9.xxx aarch64

# Check available memory
free -h
# Note: You should have at least 1GB free before starting

# Check disk space
df -h /
# Note: Need at least 2GB free for test outputs

# Check GPU status (Jetson-specific)
sudo tegrastats
# Press Ctrl+C after a few seconds
# You should see RAM, CPU, GPU stats

# Check if jtop is available
jtop --version
# If not installed: sudo -H pip3 install jetson-stats

# Check Docker (for containerized deployment)
docker --version
docker-compose --version
```

---

## Docker Testing (Recommended)

If using Docker deployment, testing is simplified:

### Build and Verify

```bash
cd ~/crustacean-monitoring

# Build the image
docker-compose build

# Verify image was created
docker images | grep crustacean
```

### Run Tests in Container

```bash
# Interactive shell for testing
docker-compose run shell

# Inside container, run tests
python -m pytest tests/ -v
python -m pytest tests/ --cov=crustacean

# Exit container
exit
```

### Test Pipelines via Docker

```bash
# Test offline pipeline
docker-compose run offline

# Test real-time (with camera)
docker-compose run realtime

# Test monitoring
docker-compose run monitor
```

### Docker-Specific Checks

```bash
# Verify GPU access in container
docker-compose run shell nvidia-smi

# Verify camera access
docker-compose run shell ls -la /dev/video*

# Check logs
docker-compose logs
```

If using native installation instead, continue with the sections below.

---

## Task 21.1: Run Full Test Suite

**Goal:** Verify all unit and integration tests pass on the Jetson Nano platform, and that test coverage meets the 70% target.

### Step 1: Deploy Code to Jetson

There are two ways to get the code onto your Jetson:

**Option A: Git Clone (Recommended)**
```bash
# SSH into Jetson
ssh jetson@<jetson-ip>

# Navigate to home directory
cd ~

# Clone the repository (first time)
git clone https://github.com/EvanON-test/crustacean-monitoring.git
cd crustacean-monitoring

# Or pull latest changes (if already cloned)
cd ~/crustacean-monitoring
git pull origin main

# Fetch large files (models)
git lfs pull
```

**Option B: SCP Transfer**
```bash
# From your development machine
scp -r /path/to/crustacean-monitoring jetson@<jetson-ip>:~/
```

> **Important:** If using SCP, make sure to include the model files. They are large and stored with Git LFS.

### Step 2: Set Up Python Environment

```bash
cd ~/crustacean-monitoring

# Check Python version (need 3.9+)
python3 --version
# If below 3.9, you may need to install it:
# sudo apt-get install python3.9 python3.9-pip

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
# Install main dependencies
# This may take 5-10 minutes on Jetson
pip install -r requirements.txt

# Install Jetson-specific packages
pip install -r requirements-jetson.txt

# Install development/testing dependencies
pip install -r requirements-dev.txt

# Verify key packages installed
pip list | grep -E "pytest|numpy|opencv|tflite"
```

**Expected packages:**
- pytest >= 7.0
- numpy >= 1.20
- opencv-python >= 4.5
- tflite-runtime >= 2.5
- psutil
- PyYAML

> **Troubleshooting:** If tflite-runtime fails to install, try:
> ```bash
> pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
> ```

### Step 4: Verify Models Are Present

```bash
# Check all model files exist and have reasonable sizes
echo "=== Binary Classifier ==="
ls -lh processing/binary_classifier/save/*.tflite
# Expected: ~2-5 MB file

echo "=== Frame Selector ==="
ls -lh processing/frame_selector/*.tflite
# Expected: Two files, ~1-3 MB each

echo "=== Object Detector ==="
ls -lh processing/object_detector/*.tflite
# Expected: ~5-20 MB file

echo "=== Keypoint Detector ==="
ls -lh processing/keypoint_detector/models/*.tflite
# Expected: ~1-5 MB file
```

> **Warning:** If any files show as ~130 bytes, Git LFS didn't download them properly. Run `git lfs pull` again.

### Step 5: Run Unit Tests

```bash
# Run all unit tests with verbose output
python -m pytest tests/unit/ -v

# If you want to see print statements during tests
python -m pytest tests/unit/ -v -s

# Run tests for a specific module (useful for debugging)
python -m pytest tests/unit/test_config.py -v
python -m pytest tests/unit/test_models/ -v
python -m pytest tests/unit/test_pipeline.py -v
```

**What to look for:**
- All tests should show `PASSED` in green
- No `FAILED` or `ERROR` in red
- No import errors at the start

**Record your results:**
```
Unit Test Results:
- Total tests: _____
- Passed: _____
- Failed: _____
- Errors: _____
- Skipped: _____

Failed tests (if any):
1. _________________________________
2. _________________________________
3. _________________________________
```

### Step 6: Run Integration Tests

Integration tests are more comprehensive and may take longer:

```bash
# Run all integration tests
python -m pytest tests/integration/ -v

# Run with timeout (in case tests hang)
python -m pytest tests/integration/ -v --timeout=300
```

> **Note:** Integration tests may require more memory. If you see OOM errors, try running tests individually:
> ```bash
> python -m pytest tests/integration/test_offline_pipeline_integration.py -v
> python -m pytest tests/integration/test_realtime_pipeline_integration.py -v
> ```

**Record your results:**
```
Integration Test Results:
- Total tests: _____
- Passed: _____
- Failed: _____
- Time taken: _____ seconds

Failed tests (if any):
1. _________________________________
2. _________________________________
```

### Step 7: Check Test Coverage

```bash
# Run all tests with coverage measurement
python -m pytest tests/ --cov=crustacean --cov-report=term-missing

# Generate HTML report (easier to read)
python -m pytest tests/ --cov=crustacean --cov-report=html
# Open htmlcov/index.html in a browser
```

**Coverage targets:**
| Module | Target | Actual |
|--------|--------|--------|
| crustacean/core/ | 75% | ____% |
| crustacean/models/ | 70% | ____% |
| crustacean/utils/ | 90% | ____% |
| crustacean/monitoring/ | 70% | ____% |
| crustacean/camera/ | 60% | ____% |
| crustacean/threads/ | 70% | ____% |
| **Overall** | **70%** | **____%** |

### Step 8: Task 21.1 Checklist

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] No import errors
- [ ] No missing dependencies
- [ ] Overall coverage >= 70%
- [ ] No critical modules with < 50% coverage

---

## Task 21.2: Test on Jetson Nano Hardware

**Goal:** Verify the system works correctly with real hardware (camera, GPU) on the Jetson Nano.

### Step 1: Camera Verification

Before testing the pipelines, verify your camera works:

**For CSI Camera:**
```bash
# Test with GStreamer (should show live video)
gst-launch-1.0 nvarguscamerasrc ! \
    'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
    nvvidconv ! \
    'video/x-raw,format=BGRx' ! \
    videoconvert ! \
    xvimagesink

# Press Ctrl+C to stop
# If you see video, camera is working!
```

**For USB Camera:**
```bash
# Check device exists
ls -la /dev/video*

# Test with simple capture
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    videoconvert ! \
    xvimagesink
```

> **Common Issues:**
> - "Cannot open camera": Check cable connection, try `sudo systemctl restart nvargus-daemon`
> - "No video": Camera may be in use by another process, run `fuser /dev/video0`

**Record camera info:**
```
Camera Type: CSI / USB
Device: /dev/video_____
Resolution tested: _____ x _____
Framerate: _____ fps
Status: Working / Not Working
```

### Step 2: Test Offline Pipeline

The offline pipeline processes pre-recorded video files. This is the safest test to start with.

**Preparation:**
```bash
# Verify test videos exist
ls -la processing/video/
# You should see .mp4 or .avi files

# If no videos, you can record one:
# gst-launch-1.0 nvarguscamerasrc num-buffers=300 ! \
#     'video/x-raw(memory:NVMM),width=1280,height=720' ! \
#     nvvidconv ! nvv4l2h264enc ! h264parse ! \
#     mp4mux ! filesink location=processing/video/test.mp4

# Clear any previous results
rm -f CompletedFiles.txt
rm -rf realtime_frames/*
```

**Run the test:**
```bash
# Run with debug logging to see detailed output
python scripts/run_offline.py \
    --video-dir ./processing/video \
    --log-level DEBUG

# Watch the output for:
# - "Loading models" messages
# - "Processing video: xxx" for each video
# - "Binary Classifier" timing
# - "Frame Selector" timing
# - "Object Detector" timing
# - "Keypoint Detector" timing
# - "Pipeline completed successfully"
```

**Monitor system resources (in another terminal):**
```bash
# Watch memory and CPU
watch -n 1 'free -h && echo "---" && top -bn1 | head -15'

# Or use jtop for comprehensive monitoring
jtop
```

**Verify outputs:**
```bash
# Check completed files log
cat CompletedFiles.txt

# Check output directory
ls -la realtime_frames/
# Or check configured output directory

# Examine a keypoint CSV file
head -5 realtime_frames/*/keypoints.csv
```

**Record results:**
```
Offline Pipeline Test:
- Videos found: _____
- Videos processed: _____
- Processing time: _____ seconds
- Peak memory usage: _____ MB
- Output files created: Yes / No
- Keypoint CSVs valid: Yes / No

Errors encountered:
_________________________________
_________________________________
```

### Step 3: Test Real-time Pipeline (Headless Mode)

This tests the camera capture and processing without display overhead.

```bash
# Run in headless mode
sudo python scripts/run_realtime.py --log-level DEBUG

# The pipeline will:
# 1. Initialize camera
# 2. Load all models
# 3. Start processing loop
# 4. Detect motion and process frames
# 5. Save detections

# To trigger detections:
# - Wave your hand in front of the camera
# - Move an object across the field of view
# - Wait for "Motion detected" log messages

# Let it run for 2-3 minutes, then press Ctrl+C
```

**What to watch for in the logs:**
```
✓ "Camera opened" - Camera initialized successfully
✓ "Models instantiated" - All 4 models loaded
✓ "Starting main processing loop" - Pipeline running
✓ "Motion detected" - Motion detection working
✓ "Collection complete: 30 frames" - Frame collection working
✓ "Detection result: frame=X, confidence=Y" - Detection working
✓ "Detection X submitted for saving" - Save working
✓ "Shutting down" - Graceful shutdown on Ctrl+C
```

**Record results:**
```
Real-time Headless Test:
- Camera initialized: Yes / No
- Models loaded: Yes / No
- Motion detection triggered: Yes / No (_____ times)
- Detections saved: Yes / No (_____ detections)
- Graceful shutdown: Yes / No
- Runtime: _____ minutes

Errors encountered:
_________________________________
_________________________________
```

### Step 4: Test Real-time Pipeline (Display Mode)

> **Requirement:** Physical monitor connected to Jetson, or X11 forwarding enabled.

```bash
# With physical monitor
sudo python scripts/run_realtime.py --display --log-level INFO

# With X11 forwarding (slower)
ssh -X jetson@<ip>
sudo python scripts/run_realtime.py --display --log-level INFO
```

**What to verify:**
1. Video window appears with camera feed
2. Overlay text shows:
   - Frame counter (incrementing)
   - Detection count
   - Runtime
   - Confidence (when detection occurs)
   - "Collecting: X/30" during frame collection
3. Press 'q' to quit - should exit cleanly

**Record results:**
```
Real-time Display Test:
- Window appeared: Yes / No
- Video feed visible: Yes / No
- Overlay text visible: Yes / No
- Frame counter updating: Yes / No
- Approximate FPS: _____ fps
- 'q' quit works: Yes / No

Visual issues:
_________________________________
```

### Step 5: Test Monitoring System

```bash
# Run monitoring with offline pipeline
python scripts/run_monitoring.py \
    --video-dir ./processing/video \
    --output benchmark/test_metrics.csv \
    --interval 1.0 \
    --log-level DEBUG

# After completion, examine the metrics
cat benchmark/test_metrics.csv
```

**Verify metrics file contains:**
```bash
# Check CSV headers
head -1 benchmark/test_metrics.csv
# Should include: timestamp, cpu_percent, ram_percent, cpu_temp, gpu_temp, etc.

# Check data rows
tail -10 benchmark/test_metrics.csv
# Should have numeric values, not all "N/A"

# Count samples
wc -l benchmark/test_metrics.csv
```

**Record results:**
```
Monitoring Test:
- Monitor started: Yes / No
- Metrics file created: Yes / No
- Samples collected: _____
- CPU temp recorded: Yes / No (range: _____°C - _____°C)
- GPU temp recorded: Yes / No (range: _____°C - _____°C)
- RAM usage recorded: Yes / No (range: _____% - _____%)

Issues:
_________________________________
```

### Step 6: Compare with Original Implementation

If you still have the original scripts available:

```bash
# Time the original pipeline
time sudo python pipeline.py

# Time the new pipeline
time python scripts/run_offline.py --video-dir ./processing/video

# Compare outputs
diff -r <original_output_dir> <new_output_dir>
```

**Record comparison:**
```
Performance Comparison:
- Original pipeline time: _____ seconds
- New pipeline time: _____ seconds
- Difference: _____% (faster/slower)

Output Comparison:
- Same number of detections: Yes / No
- Keypoint values similar: Yes / No
- Any missing outputs: _________________________________
```

### Step 7: Task 21.2 Checklist

- [ ] Camera verified working
- [ ] Offline pipeline processes videos successfully
- [ ] Real-time headless mode works
- [ ] Real-time display mode works
- [ ] Monitoring system collects metrics
- [ ] Performance comparable to original (within 20%)
- [ ] No crashes or hangs
- [ ] Graceful shutdown works

---

## Task 21.3: Performance Benchmarking

**Goal:** Measure and document the performance characteristics of the system.

### Step 1: Profiling Setup

```bash
# Ensure system is in a clean state
sudo systemctl stop unnecessary-services  # if any
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches  # Clear caches

# Check current system load
uptime
# Load average should be < 1.0 before starting
```

### Step 2: Run Profiled Offline Pipeline

```bash
# Run with profiling enabled
python scripts/run_offline.py \
    --video-dir ./processing/video \
    --profile \
    --log-level INFO

# The profiler will output a summary like:
# ======================================================================
# PERFORMANCE PROFILE: offline_pipeline
# Total runtime: XX.XXs
# ======================================================================
# Section                   Count    Total      Mean       Min        Max
# ----------------------------------------------------------------------
# Binary Classifier            X     XX.XXXs   XXX.XXms   XXX.XXms   XXX.XXms
# Frame Selector               X     XX.XXXs   XXX.XXms   XXX.XXms   XXX.XXms
# Object Detector              X     XX.XXXs   XXX.XXms   XXX.XXms   XXX.XXms
# Keypoint Detector            X     XX.XXXs   XXX.XXms   XXX.XXms   XXX.XXms
# ======================================================================
```

**Record profiling results:**
```
Profiling Results (Offline Pipeline):

Total Runtime: _____ seconds
Number of videos: _____

Stage Timings:
| Stage | Count | Total (s) | Mean (ms) | Min (ms) | Max (ms) |
|-------|-------|-----------|-----------|----------|----------|
| Binary Classifier | | | | | |
| Frame Selector | | | | | |
| Object Detector | | | | | |
| Keypoint Detector | | | | | |
```

### Step 3: Memory Usage Analysis

```bash
# Terminal 1: Start memory monitoring
while true; do
    echo "$(date +%H:%M:%S) $(free -m | grep Mem | awk '{print $3}')" >> memory_log.txt
    sleep 1
done

# Terminal 2: Run pipeline
python scripts/run_offline.py --video-dir ./processing/video

# Terminal 1: Stop monitoring (Ctrl+C)

# Analyze memory log
echo "Memory usage over time:"
cat memory_log.txt

# Find peak
echo "Peak memory: $(sort -t' ' -k2 -n memory_log.txt | tail -1)"
```

**Record memory results:**
```
Memory Analysis:
- Baseline (before pipeline): _____ MB
- Peak (during pipeline): _____ MB
- After pipeline: _____ MB
- Memory increase: _____ MB
- Memory leaked: _____ MB (should be ~0)
```

### Step 4: Real-time Performance Metrics

```bash
# Run real-time with profiling for 5 minutes
timeout 300 sudo python scripts/run_realtime.py --profile --log-level INFO

# Or manually stop after 5 minutes with Ctrl+C
```

**During the test, trigger motion periodically** (every 30 seconds or so) to get detection metrics.

**Record results:**
```
Real-time Performance (5 minute test):
- Total frames captured: _____
- Frames per second: _____ fps
- Motion triggers: _____
- Successful detections: _____
- Failed detections: _____
- Average detection latency: _____ seconds
  (time from motion to saved file)

Frame Capture Timing:
- Mean: _____ ms
- Min: _____ ms
- Max: _____ ms
```

### Step 5: Thermal Performance

```bash
# Monitor temperatures during extended run
# Terminal 1: Temperature logging
while true; do
    TEMPS=$(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | tr '\n' ' ')
    echo "$(date +%H:%M:%S) $TEMPS" >> thermal_log.txt
    sleep 5
done

# Terminal 2: Run pipeline for 10 minutes
timeout 600 python scripts/run_offline.py --video-dir ./processing/video --profile

# Analyze thermal log
echo "Temperature readings (millidegrees C):"
cat thermal_log.txt
```

**Record thermal results:**
```
Thermal Analysis (10 minute run):
- Starting CPU temp: _____°C
- Peak CPU temp: _____°C
- Ending CPU temp: _____°C
- Starting GPU temp: _____°C
- Peak GPU temp: _____°C
- Ending GPU temp: _____°C
- Thermal throttling observed: Yes / No
```

> **Warning:** If temperatures exceed 80°C, consider adding cooling or reducing processing load.

### Step 6: Stress Test

Run multiple iterations to check for stability:

```bash
# Run 5 iterations
for i in 1 2 3 4 5; do
    echo "========== Run $i of 5 =========="
    
    # Clear previous results
    rm -f CompletedFiles.txt
    
    # Run pipeline
    time python scripts/run_offline.py \
        --video-dir ./processing/video \
        --profile
    
    # Check memory after each run
    echo "Memory after run $i:"
    free -h
    
    # Brief pause between runs
    sleep 10
done
```

**Record stress test results:**
```
Stress Test (5 iterations):

| Run | Time (s) | Peak Mem (MB) | Errors |
|-----|----------|---------------|--------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

Consistency: Times within ±10%? Yes / No
Memory leaks: Increasing memory trend? Yes / No
Stability: All runs completed? Yes / No
```

### Step 7: Task 21.3 Checklist

- [ ] Profiling data collected for all pipeline stages
- [ ] Memory usage documented
- [ ] Real-time FPS measured
- [ ] Detection latency measured
- [ ] Thermal behavior documented
- [ ] Stress test completed (5 runs)
- [ ] No memory leaks detected
- [ ] Performance is acceptable for use case

---

## Troubleshooting Guide

### Camera Issues

**Problem: "Cannot open camera" or "Camera initialization failed"**
```bash
# Check if camera is detected
ls /dev/video*

# Check if another process is using the camera
fuser /dev/video0

# Kill any process using the camera
sudo fuser -k /dev/video0

# Restart the camera daemon (CSI cameras)
sudo systemctl restart nvargus-daemon

# Check kernel messages for camera errors
dmesg | tail -20
```

**Problem: "GStreamer error" or "Pipeline failed"**
```bash
# Test basic GStreamer
gst-launch-1.0 videotestsrc ! videoconvert ! xvimagesink

# If that works, test camera source
gst-launch-1.0 nvarguscamerasrc ! fakesink

# Check GStreamer plugins
gst-inspect-1.0 nvarguscamerasrc
```

### Memory Issues

**Problem: "Out of memory" or system becomes unresponsive**
```bash
# Check current memory
free -h

# Add swap space (temporary)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Reduce memory usage in config
# Edit config/default_config.yaml:
# realtime:
#   frames_to_collect: 20  # Reduce from 30
#   max_save_threads: 1    # Reduce from 2
```

**Problem: Memory keeps increasing (leak)**
```bash
# Run with memory profiling
pip install memory_profiler
python -m memory_profiler scripts/run_offline.py --video-dir ./processing/video
```

### Model Issues

**Problem: "Model not found" or "Failed to load model"**
```bash
# Verify model files
find processing/ -name "*.tflite" -exec ls -lh {} \;

# Re-download with Git LFS
git lfs pull

# Check file integrity
file processing/binary_classifier/save/*.tflite
# Should say "data" not "ASCII text"
```

**Problem: "TFLite interpreter error"**
```bash
# Check TFLite version
python -c "import tflite_runtime; print(tflite_runtime.__version__)"

# Try reinstalling
pip uninstall tflite-runtime
pip install tflite-runtime==2.13.0
```

### Performance Issues

**Problem: Very slow processing**
```bash
# Check CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Should be "performance" not "powersave"

# Set to performance mode
sudo nvpmodel -m 0  # Max performance on Jetson
sudo jetson_clocks  # Lock clocks to max

# Check for thermal throttling
cat /sys/devices/virtual/thermal/thermal_zone*/temp
# If > 80000 (80°C), add cooling
```

**Problem: Low FPS in real-time mode**
```bash
# Increase process interval (check less frequently)
# Edit config or use environment variable:
export CRUSTACEAN_REALTIME_PROCESS_INTERVAL=60

# Reduce resolution
# Edit config:
# camera:
#   width: 640
#   height: 480
```

### jtop/Monitoring Issues

**Problem: "jtop not found" or monitoring shows N/A**
```bash
# Install jetson-stats
sudo -H pip3 install -U jetson-stats

# Restart the service
sudo systemctl restart jtop.service

# If still not working, reboot
sudo reboot

# After reboot, test
jtop
```

---

## Results Template

Copy this template and fill in your results:

```markdown
# Crustacean Monitoring System - Test Results

## Test Environment

- **Date:** _______________
- **Tester:** _______________
- **Jetson Model:** Jetson Nano ___GB
- **JetPack Version:** _______________
- **Python Version:** _______________
- **Camera:** CSI / USB (model: _______________)

## Task 21.1: Test Suite Results

### Unit Tests
- Total: _____
- Passed: _____
- Failed: _____
- Coverage: _____%

### Integration Tests
- Total: _____
- Passed: _____
- Failed: _____

### Failed Tests (if any)
1. 
2. 
3. 

**Status:** ✅ PASS / ❌ FAIL

---

## Task 21.2: Hardware Testing Results

### Camera Test
- Type: CSI / USB
- Status: ✅ Working / ❌ Not Working

### Offline Pipeline
- Videos processed: _____/_____
- Time: _____ seconds
- Outputs created: ✅ Yes / ❌ No
- Status: ✅ PASS / ❌ FAIL

### Real-time Headless
- Camera init: ✅ Yes / ❌ No
- Motion detection: ✅ Working / ❌ Not Working
- Detections saved: _____
- Graceful shutdown: ✅ Yes / ❌ No
- Status: ✅ PASS / ❌ FAIL

### Real-time Display
- Window appeared: ✅ Yes / ❌ No
- Overlay visible: ✅ Yes / ❌ No
- FPS: ~_____ fps
- 'q' quit works: ✅ Yes / ❌ No
- Status: ✅ PASS / ❌ FAIL

### Monitoring
- Metrics collected: _____
- CPU temp: _____°C - _____°C
- GPU temp: _____°C - _____°C
- Status: ✅ PASS / ❌ FAIL

**Overall Hardware Status:** ✅ PASS / ❌ FAIL

---

## Task 21.3: Performance Results

### Offline Pipeline Timing
| Stage | Mean (ms) | Total (s) |
|-------|-----------|-----------|
| Binary Classifier | | |
| Frame Selector | | |
| Object Detector | | |
| Keypoint Detector | | |
| **Total** | | |

### Memory Usage
- Baseline: _____ MB
- Peak: _____ MB
- Leaked: _____ MB

### Real-time Performance
- Frame rate: _____ fps
- Detection latency: _____ seconds

### Thermal
- Peak CPU: _____°C
- Peak GPU: _____°C
- Throttling: ✅ Yes / ❌ No

### Stress Test (5 runs)
- All completed: ✅ Yes / ❌ No
- Consistent timing: ✅ Yes / ❌ No
- Memory stable: ✅ Yes / ❌ No

**Performance Status:** ✅ PASS / ❌ FAIL

---

## Overall Assessment

| Task | Status |
|------|--------|
| 21.1 Test Suite | ✅ / ❌ |
| 21.2 Hardware Testing | ✅ / ❌ |
| 21.3 Performance | ✅ / ❌ |

**Final Status:** ✅ READY FOR PRODUCTION / ❌ NEEDS FIXES

### Issues Found
1. 
2. 
3. 

### Recommendations
1. 
2. 
3. 

### Sign-off
- Tested by: _______________
- Date: _______________
- Signature: _______________
```

---

## Quick Reference Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=crustacean

# Offline pipeline
python scripts/run_offline.py --video-dir ./processing/video --profile

# Real-time headless
sudo python scripts/run_realtime.py --log-level DEBUG

# Real-time with display
sudo python scripts/run_realtime.py --display

# Monitoring
python scripts/run_monitoring.py --video-dir ./processing/video --output metrics.csv

# Check system resources
htop
jtop
free -h
df -h

# Check temperatures
cat /sys/devices/virtual/thermal/thermal_zone*/temp
```
