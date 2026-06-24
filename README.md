# LLMs as Hardware Critics: Line-of-Code Level Prediction of Hardware Design Quality from Verilog Code

Modern chips design is extremely complex, and there is a crucial need for early-stage prediction of key design-quality metrics like timing and routing congestion directly from Verilog code (a commonly used programming language for hardware design). It is especially important yet complex to predict on individual lines of code that may cause timing violations or downstream routing congestion. Prior solutions first convert Verilog to an intermediate graph representation, thereby losing rich semantic information, and only predict module-level but not line-level congestion and timing. Here, we propose **VeriLoC**, the first method that predicts design quality directly from Verilog at both the line- and module-level. To this end, VeriLoC leverages recent Verilog code-generation LLMs to extract local line-level and module-level embeddings, and trains down-stream classifiers/regressors on concatenations of these embeddings. VeriLoC achieves high F1-scores of **0.86–0.95** for line-level congestion and timing prediction, and reduces the mean average percentage error from **14% − 18%** for current SOTA methods down to only 4%. We believe that VeriLoC embeddings and several insights from our work will be of value for other predictive and optimization tasks for complex hardware design.

# Dataset
To create dataset we used Synopsys RTL-A tool, which provides line level QoR metric. To run RTL-A tool, follow the provided reference methodology with the tool. This step is to create labels. It can be replaced with any other similar tool. For module level prediction, synthesis tools like Synopsys DC or open-source yosys tool can be used.

# Training
## Steps to reproduce the results
- Install the dependencies using `requirements.txt`
- Generate the pre-computed CLVerilog embeddings of the dataset
- Train the autoencoder
- Train Congestion Classifier
- Train Timing Classifier
- Train Timing Regressor

## Generate CLVerilog Embeddings
- Replace the `HUGGING_FACE_TOKEN` in `generate_embeddings.sh`
- Run `generate_embeddings.sh`:

```bash
sh generate_embeddings.sh
```

- This generates both module and line level embeddings. The respective code parts can be commented out if not needed.
- The module embeddings are saved to `embeddings/hidden_states_modules.sh` and the line level embeddings are saved to `embeddings/openROAD_low_level_modules_yosys_v1_embeddings` directory with the same directory structure as the dataset.

## Train Autoencoder
- To train the autoencoder, run the command:

```bash
python train/autoencoder.py
```

## Train Congestion Classifier
- To train the congestion classifier, run the command:

```bash
python train/congestion.py
```

## Train Timing Classifier
- To train the timing classifier, run the command:

```bash
python train/timing.py
```

# Inference

## Detect Congestion causing lines in RTL

```bash
python inference/congestion.py <verilog file path>
```

## Detect Timing causing lines and WNS in RTL

```bash
python inference/timing.py <verilog file path>
```

# Saliency Map Interpretability
- To generate the saliency interpretations of the RTL as shown in the paper, please follow the self explanatory `visualization/visualization.ipynb` jupyter notebook.
- A `HUGGINGFACE TOKEN` has to be added in the jupyter notebook.


## Citing VeriLoC

If you use VeriLoC in your research or project, please cite it using the following BibTeX entry:

```bibtex
@inproceedings{NEURIPS2025_141ad4ee,
 author = {Hemadri, Raghu Vamshi and Bhandari, Jitendra and Nakkab, Andre and Knechtel, Johann and Gopalan, Badri and Narayanaswamy, Ramesh and Karri, Ramesh and Garg, Siddharth},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
 pages = {13541--13568},
 publisher = {Curran Associates, Inc.},
 title = {VeriLoC: Line-of-Code Level Prediction of Hardware Design Quality from Verilog Code},
 url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/141ad4ee52ba0b1db87483cb8f173c4e-Paper-Conference.pdf},
 volume = {38},
 year = {2025}
}
