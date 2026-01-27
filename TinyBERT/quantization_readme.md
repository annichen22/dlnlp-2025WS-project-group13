# TinyBERT Quantization Guide

This document explains the quantization process applied to TinyBERT models, the use of ONNX for benchmarking, and the technical details of the quantization methods employed.

---

## Table of Contents

1. [Overview](#overview)
2. [Why Use ONNX for Benchmarking](#why-use-onnx-for-benchmarking)
3. [Quantization Process Workflow](#quantization-process-workflow)
4. [Quantization Methods Explained](#quantization-methods-explained)
5. [Benchmarking Methodology](#benchmarking-methodology)
6. [Results Interpretation](#results-interpretation)

---

## Overview

**Quantization** is a model compression technique that converts high-precision floating-point weights and activations (typically FP32) to lower-precision integer representations (typically INT8). This process offers several benefits:

- **Reduced Model Size**: INT8 models are approximately 4× smaller than FP32 models
- **Faster Inference**: Integer arithmetic is faster than floating-point operations on most hardware
- **Lower Memory Bandwidth**: Smaller data types reduce memory transfer overhead
- **Energy Efficiency**: Integer operations consume less power

This project applies quantization to both the teacher (BERT-base) and distilled student (TinyBERT) models to evaluate the trade-offs between model compression, inference speed, and task accuracy.

---

## Why Use ONNX for Benchmarking

### The Challenge of Fair Comparison

When comparing models with different architectures (teacher vs. student) and different optimization techniques (distillation vs. quantization), we need a **standardized inference runtime** to ensure fair benchmarking. PyTorch models can have varying performance characteristics due to:

- Different framework optimizations
- Inconsistent kernel implementations
- Variations in graph execution strategies

### ONNX Runtime Benefits

**ONNX (Open Neural Network Exchange)** provides:

1. **Hardware-agnostic representation**: Models are converted to a standardized intermediate format
2. **Optimized inference engine**: ONNX Runtime applies graph optimizations and uses hardware-specific kernels
3. **Consistent execution**: All models run through the same execution pipeline
4. **Built-in quantization support**: Native support for quantized model inference with optimized INT8 operators
5. **Cross-platform benchmarking**: Enables reproducible performance measurements across different hardware

### Our Benchmarking Setup

```python
# All models are converted to ONNX format
teacher_onnx = export_bert_to_onnx(teacher_model_dir)
student_onnx = export_bert_to_onnx(student_model_dir)
quantized_onnx = quantize_static(teacher_onnx, ...)

# Fair comparison using ONNX Runtime with identical configurations
providers = ["CPUExecutionProvider"]
results = {
    "teacher": evaluate_onnx(teacher_onnx, providers),
    "student": evaluate_onnx(student_onnx, providers),
    "quantized": evaluate_onnx(quantized_onnx, providers),
}
```

This ensures that performance differences reflect actual model characteristics rather than framework artifacts.

---

## Quantization Process Workflow

### Step 1: Export PyTorch Models to ONNX

Models are first converted from PyTorch to ONNX format using `torch.onnx.export()`:

```python
export_bert_to_onnx(
    teacher_model_dir,
    out_path="model.onnx",
    seq_len=128,
    opset=17,  # ONNX opset version
)
```

**Key parameters**:
- `seq_len=128`: Maximum sequence length for input tokenization
- `opset=17`: ONNX operator set version (must be ≥10 for quantization support)
- Dynamic axes for batch size and sequence length

### Step 2: Prepare Calibration Data

Static quantization requires a **calibration dataset** to compute quantization parameters:

```python
# Use subset of training data
calibration_examples = train_examples[:512]
features = convert_examples_to_features(
    calibration_examples,
    label_list,
    max_seq_length=128,
    tokenizer,
    output_mode
)
```

The calibration data should be representative of the inference distribution to ensure optimal quantization parameters.

### Step 3: Apply Static Quantization

Two quantization configurations are applied:

#### Configuration 1: QDQ Format with QInt8
```python
quantize_static(
    model_input=teacher_model_onnx_path,
    model_output=quantized_model_path_1,
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.MinMax,
)
```

#### Configuration 2: QOperator Format with QUInt8/QInt8
```python
quantize_static(
    model_input=teacher_model_onnx_path,
    model_output=quantized_model_path_2,
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QOperator,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.MinMax,
)
```

### Step 4: Evaluate and Benchmark

All models (teacher, students, quantized) are evaluated on the same test set using ONNX Runtime:

- **Task Performance**: Accuracy/F1/correlation on GLUE tasks
- **Model Size**: File size in MB
- **Inference Time**: Mean latency per batch with warmup

---

## Quantization Methods Explained

### Static Quantization

**Static quantization** pre-computes quantization parameters (scale and zero-point) using calibration data. These parameters remain fixed during inference.

#### Mathematical Foundation

Quantization maps floating-point values to 8-bit integers using a linear transformation:

```
val_fp32 = scale × (val_quantized - zero_point)
```

Where:
- **scale**: Positive real number mapping FP32 range to INT8 range
- **zero_point**: Integer representing zero in quantized space

For **symmetric quantization** (centered at zero):
```
scale = max(|data_max|, |data_min|) × 2 / (quant_max - quant_min)
zero_point = 0
```

For **asymmetric quantization**:
```
scale = (data_max - data_min) / (quant_max - quant_min)
zero_point = quant_min - round(data_min / scale)
```

### Quantization Formats

#### QDQ (Quantize-DeQuantize) Format

**Concept**: Inserts `QuantizeLinear` and `DeQuantizeLinear` operators around original FP32 operators.

**Graph structure**:
```
Input (FP32) → QuantizeLinear → INT8 Tensor → DeQuantizeLinear → FP32 → Operator
```

**Characteristics**:
- Preserves original operator semantics
- Easier to debug (can inspect quantized values)
- Compatible with Quantization-Aware Training (QAT)
- Better for GPUs and hardware without native INT8 support
- Used in our **Configuration 1**

**Advantages**:
- More flexible: can selectively quantize specific layers
- Compatible with mixed-precision inference
- Easier to integrate with existing FP32 optimizations

#### QOperator Format

**Concept**: Replaces FP32 operators with specialized quantized operators (e.g., `QLinearConv`, `MatMulInteger`).

**Graph structure**:
```
Input (INT8) → QLinearMatMul (INT8) → Output (INT8)
```

**Characteristics**:
- Native INT8 operators throughout the graph
- More efficient on CPUs with VNNI (Vector Neural Network Instructions)
- Typically faster inference on x86-64 CPUs
- Used in our **Configuration 2**

**Advantages**:
- Maximum performance on supported hardware
- Lower memory footprint (stays in INT8 domain)
- Reduced quantization/dequantization overhead

### Activation and Weight Types

#### QInt8 (Signed 8-bit Integer)

- **Range**: -128 to 127
- **Use case**: Weights and activations
- **Format**: `S8S8` (signed activations, signed weights)
- **Used in**: Configuration 1

**Advantages**:
- Symmetric quantization is natural for signed types
- Better for data centered around zero (common in neural networks after normalization)

#### QUInt8 (Unsigned 8-bit Integer)

- **Range**: 0 to 255
- **Use case**: Activations (when always positive, e.g., after ReLU)
- **Format**: `U8S8` (unsigned activations, signed weights)
- **Used in**: Configuration 2

**Advantages**:
- Slightly better dynamic range for non-negative activations
- Default format for many quantization frameworks
- Compatible with AVX2/AVX512 `VPMADDUBSW` instruction on x86-64

### Calibration Methods

#### MinMax Calibration

**Used in both configurations**, MinMax is the simplest calibration method:

```python
scale = (max(activations) - min(activations)) / (quant_max - quant_min)
```

**Process**:
1. Run calibration data through the model
2. Record minimum and maximum values for each activation tensor
3. Compute scale and zero-point based on observed range

**Characteristics**:
- **Fast**: Single pass through calibration data
- **Simple**: No complex statistics required
- **Conservative**: Uses full observed range
- **Sensitive to outliers**: A single extreme value can reduce quantization resolution

**When to use**:
- Initial quantization experiments
- Models with well-bounded activations
- When calibration time is limited

**Alternative methods** (not used in this project):
- **Entropy (KL-Divergence)**: Minimizes information loss between FP32 and INT8 distributions
- **Percentile**: Clips outliers by using percentile values instead of absolute min/max

### Why Two Configurations?

Testing both QDQ and QOperator formats allows us to:

1. **Compare accuracy**: QDQ sometimes preserves accuracy better due to per-layer flexibility
2. **Benchmark performance**: QOperator typically runs faster on CPU
3. **Evaluate hardware compatibility**: Different formats perform better on different hardware

**Expected outcomes**:
- **Configuration 1 (QDQ/QInt8)**: Better accuracy, moderate speed
- **Configuration 2 (QOperator/QUInt8/QInt8)**: Better speed on x86-64 CPUs, possible accuracy trade-off

---

## Benchmarking Methodology

### Metrics Collected

#### 1. Task Performance
- **Classification tasks** (CoLA, SST-2, MRPC, RTE): Accuracy, F1-score
- **Regression tasks** (STS-B): Pearson/Spearman correlation
- **Eval loss**: Cross-entropy or MSE loss on validation set

#### 2. Model Size
```python
model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
```
- Measured in megabytes
- Quantized models expected to be ~4× smaller than FP32

#### 3. Inference Time
```python
benchmark_onnx_model(
    onnx_path,
    eval_dataloader,
    providers=["CPUExecutionProvider"],
    warmup_runs=5,   # Stabilize performance
    num_runs=30      # Statistical reliability
)
```

**Statistics reported**:
- Mean latency (ms)
- Standard deviation
- Min/Max latency
- Median latency

### Benchmarking Best Practices

1. **Warmup runs**: First 5 iterations discarded to allow kernel compilation and cache warming
2. **Multiple runs**: 30 iterations for statistical confidence
3. **Consistent hardware**: All models benchmarked on same CPU with same provider
4. **Controlled environment**: No concurrent processes during benchmarking
5. **Batch size**: Fixed batch size across all models for fair comparison

---


### Analyzing Results

Check the generated `results.json` and `results.csv` files for each task:

```json
{
  "teacher": {
    "accuracy": 0.85,
    "eval_loss": 0.42,
    "model_size": 438.5,
    "mean_ms": 45.2
  },
  "quant_1": {
    "accuracy": 0.84,
    "eval_loss": 0.45,
    "model_size": 110.3,
    "mean_ms": 28.7
  }
}
```

**Key insights**:
- **Compression ratio** = teacher_size / quantized_size
- **Speedup** = teacher_mean_ms / quantized_mean_ms
- **Accuracy drop** = teacher_accuracy - quantized_accuracy

---

## Running the Quantization Experiments

### Prerequisites

```bash
pip install onnx onnxruntime torch transformers
```

### Execution

```python
python quantization_experiments.py
```

This will:
1. Export all models to ONNX format
2. Generate two quantized versions of the teacher model
3. Evaluate all models on the specified GLUE task
4. Benchmark inference time
5. Save results to JSON/CSV files

### Output Files

For each task (e.g., RTE):
- `RTE/ts_teacher_bert/model.onnx`: Teacher model in ONNX format
- `RTE/ts_teacher_bert/model_quant_1.onnx`: QDQ quantized model
- `RTE/ts_teacher_bert/model_quant_2.onnx`: QOperator quantized model
- `RTE/results.json`: Detailed metrics
- `RTE/results.csv`: Tabular summary

---

## References

- [ONNX Runtime Quantization Documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [ONNX: Open Neural Network Exchange](https://onnx.ai/)