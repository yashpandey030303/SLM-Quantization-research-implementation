# Small Language Model Quantization Research

## Overview

This research project investigates the impact of 4-bit quantization on Small Language Models (SLMs) performance, focusing on natural language inference tasks across different languages. The study compares baseline models against their 4-bit quantized counterparts using the XNLI (Cross-lingual Natural Language Inference) dataset.

## Research Objectives

- **Primary Goal**: Evaluate the trade-offs between model performance and resource efficiency when applying 4-bit quantization to SLMs
- **Secondary Goals**: 
  - Analyze cross-lingual performance degradation patterns
  - Measure memory consumption reduction benefits
  - Assess inference latency changes

## Models Evaluated

1. **Microsoft Phi-3-mini-4k-instruct**: A 3.8B parameter instruction-tuned model
2. **Qwen1.5-1.8B-Chat**: A 1.8B parameter conversational model

## Dataset

- **XNLI (Cross-lingual Natural Language Inference)**
- **Languages**: English (EN) and Hindi (HI)
- **Sample Size**: 50 examples per language
- **Task**: Three-way classification (entailment, neutral, contradiction)

## Methodology

### Quantization Configuration
- **Method**: BitsAndBytesConfig 4-bit quantization
- **Compute dtype**: torch.bfloat16
- **Implementation**: Hugging Face Transformers + BitsAndBytes

### Evaluation Metrics
1. **Accuracy (%)**: Percentage of correctly classified examples
2. **Average Latency (s/ex)**: Mean inference time per example
3. **Peak VRAM (MB)**: Maximum GPU memory consumption

### Experimental Setup
- **Environment**: Google Colab with GPU acceleration
- **Framework**: PyTorch + Hugging Face Transformers
- **Generation Parameters**: max_new_tokens=5, use_cache=False

## Results Summary

| Model | Type | Language | Accuracy (%) | Avg Latency (s/ex) | Peak VRAM (MB) |
|-------|------|----------|-------------|-------------------|----------------|
| Qwen1.5-1.8B-Chat | Baseline | EN | 66.00 | 0.6516 | 3531.13 |
| Qwen1.5-1.8B-Chat | Baseline | HI | 36.00 | 0.9126 | 3545.73 |
| Qwen1.5-1.8B-Chat | 4-bit Quantized | EN | 40.00 | 0.7567 | 1891.77 |
| Qwen1.5-1.8B-Chat | 4-bit Quantized | HI | 38.00 | 1.0509 | 1896.94 |
| Phi-3-mini-4k-instruct | Baseline | EN | 88.00 | 1.4115 | 7318.03 |
| Phi-3-mini-4k-instruct | Baseline | HI | 50.00 | 2.6192 | 7348.99 |
| Phi-3-mini-4k-instruct | 4-bit Quantized | EN | 88.00 | 1.6014 | 2436.02 |
| Phi-3-mini-4k-instruct | 4-bit Quantized | HI | 38.00 | 2.1491 | 2444.65 |

## Key Findings

### Memory Efficiency
- **Qwen1.5-1.8B**: ~46% VRAM reduction (3531→1892 MB)
- **Phi-3-mini**: ~67% VRAM reduction (7318→2436 MB)
- Larger models show greater absolute and relative memory savings

### Performance Impact
- **Phi-3-mini**: Maintains English performance (88%) but significant Hindi degradation (50%→38%)
- **Qwen1.5**: Notable English performance drop (66%→40%) with minimal Hindi impact (36%→38%)
- Cross-lingual robustness varies significantly between models

### Latency Changes
- **Mixed Results**: Some models show slight latency increases, others remain stable
- Quantization doesn't guarantee faster inference in all cases
- Model-specific optimizations may be required

## Technical Implementation

### Dependencies
```bash
pip install accelerate>=0.28.0
pip install transformers datasets bitsandbytes
pip install torch pandas
```

### Usage
```python
# Load quantized model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="cuda"
)
```

## Implications

### For Practitioners
- **Memory-Constrained Environments**: 4-bit quantization enables deployment of larger models on resource-limited hardware
- **Performance Sensitivity**: Critical applications should validate quantized model performance on target tasks
- **Language-Specific Considerations**: Cross-lingual performance may degrade disproportionately

### For Researchers
- **Model Architecture Dependency**: Quantization impact varies significantly across model families
- **Evaluation Methodology**: Comprehensive evaluation should include multiple languages and tasks
- **Optimization Opportunities**: Model-specific quantization strategies may yield better trade-offs

## Limitations

1. **Limited Sample Size**: 50 examples per language may not capture full performance spectrum
2. **Task Specificity**: Results specific to natural language inference tasks
3. **Language Coverage**: Only English and Hindi evaluated
4. **Hardware Dependency**: Results may vary across different GPU architectures

## Future Work

- Expand evaluation to additional languages and model sizes
- Investigate task-specific quantization strategies
- Compare with other compression techniques (pruning, distillation)
- Develop adaptive quantization methods based on cross-lingual robustness

## Files Description

- `Final_SLM_Quantisation.ipynb`: Main experimental notebook
- `slm_quantization_results.csv`: Raw experimental results
- `README.md`: This documentation file

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{slm_quantization_2024,
  title={Impact of 4-bit Quantization on Small Language Models: A Cross-lingual Analysis},
  author={[Your Name]},
  year={2024},
  note={Research Project on SLM Quantization}
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Acknowledgments

- Hugging Face for providing the transformers library and model hosting
- Microsoft and Alibaba for open-sourcing the Phi-3 and Qwen models respectively
- The XNLI dataset contributors for enabling cross-lingual evaluation
