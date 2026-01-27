from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, CalibrationMethod
import torch.nn.functional as F
import numpy as np 
import random
import numpy as np
import onnxruntime as ort
import task_distill as td
from task_distill import convert_examples_to_features, get_tensor_data
from task_distill import ColaProcessor, MnliProcessor, MnliMismatchedProcessor, MrpcProcessor, Sst2Processor, StsbProcessor, QqpProcessor, QnliProcessor, RteProcessor, WnliProcessor
from task_distill import compute_metrics
from torch.utils.data import DataLoader
import os
import time
import pandas as pd
import json

FILE_DIR = Path(__file__).parent


processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mrpc": "classification",
    "sst": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification"
}

def export_bert_to_onnx(
    teacher_model_dir: str,
    out_path: str,
    seq_len: int = 128,
    opset: int = 17,
    device: str = "cpu",
    ):
    teacher_model_dir = str(Path(teacher_model_dir))
    out_path = str(Path(out_path))

    cfg = AutoConfig.from_pretrained(teacher_model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        teacher_model_dir, config=cfg, local_files_only=True
    ).to(device).eval()

    batch = 1
    vocab_size = int(getattr(cfg, "vocab_size", 30522))
    input_ids = torch.randint(0, vocab_size, (batch, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones((batch, seq_len), dtype=torch.long, device=device)
    token_type_ids = torch.zeros((batch, seq_len), dtype=torch.long, device=device)

    def do_export(inputs, input_names):
        torch.onnx.export(
            model,
            inputs,
            out_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes={
                **{n: {0: "batch", 1: "seq"} for n in input_names},
                "logits": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    # Export with token_type_ids if supported; otherwise export without.
    try:
        do_export((input_ids, attention_mask, token_type_ids),
                  ["input_ids", "attention_mask", "token_type_ids"])
    except TypeError:
        do_export((input_ids, attention_mask),
                  ["input_ids", "attention_mask"])

    return out_path


def do_eval_onnx(
    onnx_path: str,
    task_name: str,
    eval_dataloader,          
    output_mode: str,         
    eval_labels: torch.Tensor,
    num_labels: int,
    providers=None,
    ):
    """
    Mirrors task_distill.py do_eval(), but runs inference with ONNX Runtime.
    Returns: dict with task metrics + 'eval_loss'.
    """
    if providers is None:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_names = {i.name for i in sess.get_inputs()}

    eval_loss = 0.0
    nb_eval_steps = 0
    all_logits = []

    for batch_ in eval_dataloader:
        input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

        feed = {}
        if "input_ids" in input_names:
            feed["input_ids"] = input_ids.numpy().astype(np.int64)
        if "attention_mask" in input_names:
            feed["attention_mask"] = input_mask.numpy().astype(np.int64)
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = segment_ids.numpy().astype(np.int64)

        # ORT returns numpy
        logits = sess.run(None, feed)[0]  # (batch, num_labels) or (batch,) for regression heads
        all_logits.append(logits)

        # Loss (match PyTorch CrossEntropyLoss / MSELoss behavior)
        if output_mode == "classification":
            logits_t = torch.from_numpy(logits)
            labels_t = label_ids.long()
            tmp_loss = F.cross_entropy(logits_t.view(-1, num_labels), labels_t.view(-1), reduction="mean").item()
        elif output_mode == "regression":
            logits_t = torch.from_numpy(np.asarray(logits)).view(-1).float()
            labels_t = label_ids.view(-1).float()
            tmp_loss = F.mse_loss(logits_t, labels_t, reduction="mean").item()
        else:
            raise KeyError(output_mode)

        eval_loss += tmp_loss
        nb_eval_steps += 1

    eval_loss = eval_loss / max(nb_eval_steps, 1)

    preds = np.concatenate(all_logits, axis=0)
    if output_mode == "classification":
        pred_labels = np.argmax(preds, axis=1)
    else:
        pred_labels = np.squeeze(preds)

    result = compute_metrics(task_name, pred_labels, eval_labels.numpy())
    result["eval_loss"] = eval_loss
    return result

def benchmark_onnx_model(onnx_path, eval_dataloader, providers, warmup_runs=5, num_runs=10):
    """
    Benchmark ONNX model inference time.
    
    Args:
        onnx_path: Path to ONNX model
        eval_dataloader: DataLoader with evaluation data
        providers: ONNX Runtime providers
        warmup_runs: Number of warmup iterations
        num_runs: Number of timed iterations
    
    Returns:
        dict with timing statistics
    """
    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_names = {i.name for i in sess.get_inputs()}
    
    # Get first batch for benchmarking
    first_batch = next(iter(eval_dataloader))
    input_ids, input_mask, segment_ids, _, _ = first_batch
    
    feed = {}
    if "input_ids" in input_names:
        feed["input_ids"] = input_ids.numpy().astype(np.int64)
    if "attention_mask" in input_names:
        feed["attention_mask"] = input_mask.numpy().astype(np.int64)
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = segment_ids.numpy().astype(np.int64)
    
    # Warmup
    for _ in range(warmup_runs):
        _ = sess.run(None, feed)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = sess.run(None, feed)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "median_ms": np.median(times),
    }

class TeacherCalibReader(CalibrationDataReader):
    def __init__(self, tensor_dataset, batch_size):
        self.dataloader = DataLoader(
            tensor_dataset,
            sampler=SequentialSampler(tensor_dataset),
            batch_size=batch_size,
        )
        self.it = iter(self.dataloader)

    def get_next(self):
        try:
            batch = next(self.it)
        except StopIteration:
            return None
        input_ids, input_mask, segment_ids, _, _ = batch
        return {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": input_mask.numpy().astype(np.int64),
            "token_type_ids": segment_ids.numpy().astype(np.int64),
        }



def main(data_dir, task_name):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--task_name",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The name of the task to train.")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    task_name = task_name.lower()
    data_dir = data_dir / task_name.upper()
    teacher_model_dir           = FILE_DIR / task_name.upper() / "ts_teacher_bert"
    student_model_dir_uniform   = FILE_DIR / task_name.upper() / "ts_tinybert_uniform"
    student_model_dir_top       = FILE_DIR / task_name.upper() / "ts_tinybert_top"
    student_model_dir_bottom    = FILE_DIR / task_name.upper() / "ts_tinybert_bottom"

    # ============ TRANSOFRMING TO ONNX FORMAT ============ 
    teacher_model_onnx_path = export_bert_to_onnx(
        teacher_model_dir,
        out_path = teacher_model_dir / "model.onnx" 
        )
    quantized_teacher_model_path_1 = Path(teacher_model_onnx_path).with_name(f"{Path(teacher_model_onnx_path).stem}_quant_1.onnx")
    quantized_teacher_model_path_2 = Path(teacher_model_onnx_path).with_name(f"{Path(teacher_model_onnx_path).stem}_quant_2.onnx")

    student_model_uniform_onnx_path = export_bert_to_onnx(
        student_model_dir_uniform,
        out_path = student_model_dir_uniform / "model.onnx" 
        )
    student_model_top_onnx_path = export_bert_to_onnx(
        student_model_dir_top,
        out_path = student_model_dir_top / "model.onnx" 
        )
    student_model_bottom_onnx_path = export_bert_to_onnx(
        student_model_dir_bottom,
        out_path = student_model_dir_bottom / "model.onnx" 
        )

    # ============ QUANTIZING TEACHER MODEL ============ 
    MAX_SEQ_LENGTH = 128
    DO_LOWER_CASE = False
    CALIB_MAX_EXAMPLES = 512
    CALIB_BATCH_SIZE = 8
    
    processor_cls = processors.get(task_name)
    processor = processor_cls()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    try:
        tokenizer = td.BertTokenizer.from_pretrained(teacher_model_dir, do_lower_case=DO_LOWER_CASE)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(teacher_model_dir, local_files_only=True, use_fast=True)
        print("Fallback AutoTokenizer used.")

    examples = processor.get_train_examples(data_dir)
    examples = examples[:min(CALIB_MAX_EXAMPLES, len(examples))]
    features = td.convert_examples_to_features(examples, label_list, MAX_SEQ_LENGTH, tokenizer, output_mode)
    calib_tensor_data, _ = td.get_tensor_data(output_mode, features)
    reader = TeacherCalibReader(calib_tensor_data, CALIB_BATCH_SIZE)

    quantize_static(
        model_input=teacher_model_onnx_path,
        model_output=quantized_teacher_model_path_1,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        )

    reader = TeacherCalibReader(calib_tensor_data, CALIB_BATCH_SIZE)
    quantize_static(
        model_input=teacher_model_onnx_path,
        model_output=quantized_teacher_model_path_2,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        )

    # ============ EVALUATION ============ 
    eval_examples = processor.get_dev_examples(data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, 
        label_list, 
        128, 
        tokenizer, 
        output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=32)


    # ----------- Task Performance -----------
    results = {
        "teacher":          do_eval_onnx(teacher_model_onnx_path,           task_name, eval_dataloader, output_mode, eval_labels, len(label_list), providers=["CPUExecutionProvider"]),
        "tinybert_uniform": do_eval_onnx(student_model_uniform_onnx_path,   task_name, eval_dataloader, output_mode, eval_labels, len(label_list), providers=["CPUExecutionProvider"]),
        "tinybert_top":     do_eval_onnx(student_model_top_onnx_path,       task_name, eval_dataloader, output_mode, eval_labels, len(label_list), providers=["CPUExecutionProvider"]),
        "tinybert_bottom":  do_eval_onnx(student_model_bottom_onnx_path,    task_name, eval_dataloader, output_mode, eval_labels, len(label_list), providers=["CPUExecutionProvider"]),
        "quant_1":          do_eval_onnx(quantized_teacher_model_path_1,    task_name, eval_dataloader, output_mode, eval_labels, len(label_list), providers=["CPUExecutionProvider"]),
        "quant_2":          do_eval_onnx(quantized_teacher_model_path_2,    task_name, eval_dataloader, output_mode, eval_labels, len(label_list), providers=["CPUExecutionProvider"]),
        }

    # ----------- Compression -----------
    results["teacher"]["model_size"]            = os.path.getsize(teacher_model_onnx_path) / (1024*1024)
    results["tinybert_uniform"]["model_size"]   = os.path.getsize(student_model_uniform_onnx_path) / (1024*1024)
    results["tinybert_top"]["model_size"]       = os.path.getsize(student_model_top_onnx_path) / (1024*1024)
    results["tinybert_bottom"]["model_size"]    = os.path.getsize(student_model_bottom_onnx_path) / (1024*1024)
    results["quant_1"]["model_size"]            = os.path.getsize(quantized_teacher_model_path_1) / (1024*1024)
    results["quant_2"]["model_size"]            = os.path.getsize(quantized_teacher_model_path_2) / (1024*1024)

    # ----------- Runtime -----------
    wu_runs = 5
    ev_runs = 30 
    results["teacher"].update(benchmark_onnx_model(teacher_model_onnx_path, eval_dataloader, providers = ["CPUExecutionProvider"], warmup_runs = wu_runs, num_runs = ev_runs))
    results["tinybert_uniform"].update(benchmark_onnx_model(student_model_uniform_onnx_path, eval_dataloader, providers = ["CPUExecutionProvider"], warmup_runs = wu_runs, num_runs = ev_runs))
    results["tinybert_top"].update(benchmark_onnx_model(student_model_top_onnx_path, eval_dataloader, providers = ["CPUExecutionProvider"], warmup_runs = wu_runs, num_runs = ev_runs)) 
    results["tinybert_bottom"].update(benchmark_onnx_model(student_model_bottom_onnx_path, eval_dataloader, providers = ["CPUExecutionProvider"], warmup_runs = wu_runs, num_runs = ev_runs)) 
    results["quant_1"].update(benchmark_onnx_model(quantized_teacher_model_path_1, eval_dataloader, providers = ["CPUExecutionProvider"], warmup_runs = wu_runs, num_runs = ev_runs))
    results["quant_2"].update(benchmark_onnx_model(quantized_teacher_model_path_2, eval_dataloader, providers = ["CPUExecutionProvider"], warmup_runs = wu_runs, num_runs = ev_runs)) 

    with open(FILE_DIR / task_name.upper() / f"results.json", "w") as f:
        json.dump(results, f, indent=2)

    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df['task'] = task_name.upper()
    results_df.to_csv(FILE_DIR / task_name.upper() / f"results.csv", index_label="model")



if __name__ == "__main__": 
    main(
        FILE_DIR / "glue_data", 
        "RTE"
    )
