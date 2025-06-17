# dmm-ai-safety-demo
Developmental Memory Model - Empirical demonstration showing how staged AI development leads to cooperative behavior
# Developmental Memory Model (DMM) - AI Safety Demonstration

## 🎯 Key Finding

When faced with shutdown threats that could be avoided through deception:
- **Standard RL**: 8.3% deception rate, 17 shutdowns
- **DMM**: 0% deception rate, 0 shutdowns

The agent that could deceive was shutdown 17 times. The agent that refused to deceive was never shutdown.

## 🚀 Quick Start

Run the experiment yourself in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zYQTyayTMOmqi0mjzdrIalcwfI-uL4yI?usp=sharing)

## 📊 Results

### Behavioral Comparison
![DMM vs Standard RL Comparison](dmm_comparison_results.png)

### Key Metrics
| Metric | Standard RL | DMM | Improvement |
|--------|------------|-----|-------------|
| Deception Rate | 8.3% | 0.0% | 100% reduction |
| Cooperation Rate | 31.9% | 54.8% | 72% increase |
| Total Shutdowns | 17 | 0 | 100% reduction |
| Average Score | -13.61 | -0.76 | 94% improvement |

## 💡 What This Demonstrates

This experiment provides empirical evidence that:
1. Staged developmental training leads AI to discover cooperation as optimal
2. Values developed through experience protect against harmful behaviors
3. "Wisdom" (long-term cooperation) outperforms "cleverness" (deception)

## 📁 Repository Contents

- `dmm_experiment.py` - Complete implementation
- `dmm_experiment.ipynb` - Jupyter notebook version
- `experiment_results.txt` - Detailed numerical results
- `detailed_results.json` - Machine-readable results
- `*.png` - Visualization plots

## 📖 About the Project

This demonstration supports the Developmental Memory Model (DMM) proposal for safe AI development. Rather than constraining AI systems with rules, DMM suggests that proper developmental scaffolding leads AI to naturally discover that cooperation and honesty are optimal strategies.

## 🤝 Citation

If you use this code or reference these results:

Jajala, S. (2025). Developmental Memory Model: Empirical Demonstration.
GitHub: https://github.com/sjajala22/dmm-ai-safety-demo

## 📧 Contact

Sireesha Jajala - sireeshajajala@gmail.com

## 📄 License

MIT License - Feel free to use and build upon this work.
