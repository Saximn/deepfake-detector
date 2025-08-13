# Deepfake Detector — Xception + Grad‑CAM + Gradio

A compact, end‑to‑end deepfake image detector with **explanations**. The project fine‑tunes **Xception** (via `timm`) on a real/fake image dataset, reports standard metrics, and serves **interpretable** predictions through a **Gradio** UI with **Grad‑CAM** heatmaps and optional **LLM‑generated rationales** (via Groq API).

> Notebooks: `train.ipynb` (training + evaluation) and `explain.ipynb` (interactive inference + explanations).

---

## ✨ Features
- **Backbone**: `timm.create_model('xception', pretrained=True, num_classes=2)`  
- **Data pipeline**: `Resize(299×299)` → `ToTensor()` → `Normalize(mean=[0.485, 0.456, 0.406], std=[...])`  
  + Training augments: `RandomHorizontalFlip`, `RandomRotation(±10°)`
- **Training setup**: `Adam(lr=1e-4)` + `CrossEntropyLoss` + `ReduceLROnPlateau` scheduler  
  Batch size `32`, up to `30` epochs (early save on best validation loss)
- **Metrics**: Accuracy, Precision, Recall, F1, + confusion matrices for **train** and **test**
- **Explainability**: `pytorch-grad-cam` (Grad‑CAM) heatmaps on predictions
- **UI**: Gradio app (`demo = gr.Interface(...)`, `demo.launch()`)
- **Optional LLM explanation**: natural‑language justification of the heatmap via **Groq** (`GROQ_API_KEY`)

---

🔒 Ethics & Limitations <br>
This project is for research and educational purposes only. Deepfake detection can be dataset‑biased and may not generalize to all manipulations or domains (e.g., compression artifacts, unseen generators, video vs. still images). Do not use as the sole basis for moderation, legal, or employment decisions.

🙏 Acknowledgements <br>
<ul> <li>timm for model zoo & training utilities</li>
<li>
pytorch‑grad‑cam for CAM visualizations</li>

<li>Gradio for quick interactive demos</li></ul>


License
BSD 3-Clause License

Copyright (c) 2025, Parvez Wijaya
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
