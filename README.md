## Setup

In order to run the code first setup conda environment:

```bash
cd real_estate_image_redaction
conda env create -f environment.yml
conda activate lama_object_removal
```

In order to run the inpainting process, download the big‑lama model weights provided in google drive link in Appendix A of the thesis. The checkpoints have to be inserted in `lama/big-lama/models` directory. To create the directory run:
```bash
mkdir -p lama/big-lama/models
```
To run GPT post‑filtering, create a `.env` file and store the OPENAI key in:

```bash
OPENAI_API_KEY="insert key"
```

In order to be able to run the code, download the `final_evaluation_dataset` and `optimization_dataset` from google drive link in Appendix A. To create placeholders for the images run: 

```bash
cd real_estate_image_redaction
mkdir final_evaluation_dataset pipeline_optimization_dataset
```
And store the images there.

Same goes for `ground_truth_dataset`, available in google drive link in Appendix A. To create placeholders for the grounttruth images run:
```bash
cd real_estate_image_redaction
mkdir ground_truth_dataset
```

---

## Repository Structure

This repository consists of multiple folders:

- **final_evaluation_dataset**, **pipeline_optimization_dataset**  
  Folders used as placeholders for running optimization pipelines and final evaluation pipeline. Images are available in google drive link in Appendix A of the thesis.

- **ground_truth_dataset**  
  Dataset used for validation in folder `validation_on_gt`. Data is available in google drive link in appendix of the thesis.

- **human_validation_test_results**  
  Contains results of human validation on final evaluation dataset.

- **pipeline_optimization**, **jupyter_notebooks**  
  Contains pipeline for process optimization runs from thesis. Python files are full batch runs; Jupyter notebooks were used in development on single images and display each intermediate step:  
  - `blip_decision_blur_inpaint.py`, `blip.ipynb` – Uses BLIP model to decide between inpainting and blurring. Related to Section 4.11.4 of thesis.  
  - `canny_inpaint_blur.py`, `canny_edge.ipynb` – Applies Canny edge detection to split up segmentation mask for inpainting and blurring. Related to Section 4.11.4 of thesis. 
  - `clip_postfiltering.py`, `clip_postfiltering.ipynb` – Uses CLIP to post-filter object detection for certain labels. Related to Section 4.11.2 of thesis. Quantitative results on ground truth can be found `validation_on_gt/clip_postfiltering`
  - `full_optimized_pipeline.py`, `full_optimized_pipeline.ipynb` – Full optimized pipeline based on experiments. Related to Section 4.12 of thesis. Quantitative results on ground truth can be found `validation_on_gt/results_final_optimization_run`
  - `full_process_first_pipeline.py`, `first_full_process.ipynb` – Initial pipeline consisting only of object detection, segmentation and inpainting step. Related to Section 4.11 of thesis.  Quantitative results on ground truth can be found `validation_on_gt/results_initial_pipeline`
  - `full_process_results_threshold_testing.py` – Experimenting with different confidence thresholds for object detection. Related to Section 4.11 of thesis. Quantitative results on ground truth can be found `validation_on_gt/results_threshold_testing`
  - `gpt_filtering.py`, `gpt_post_filtering.ipynb` – Experimenting with LLM as judge for post-filtering on object detection. Related to Section 4.11.2 of thesis.  Quantitative results on ground truth can be found `validation_on_gt/gpt_post_filtering_results`
  - `inpainting_only.py` – Focuses solely on the inpainting process without other pipeline stages. Related to Section 4.11.3 of thesis.  
  - `results_segmentation_dilation.py` – Adds segmentation and dilation post-processing to pipeline results. Related to Section 4.11.3 of thesis.  

- **final_pipeline_on_unseen_data**  
  - `final_evaluation_run.py` – Identical steps as in `full_optimized_pipeline.py`, but run on unseen data.

- **lama**  
  Folder contains implementation to run LaMa inpainting model adopted from:  
  https://github.com/advimman/lama

- **Result folders**  
  - `results_on_unseen_dataset`  
  - `results_pipeline_optimization_dataset`  
  - `inpaint_blur_results`  
  - `post_filtering_results`

- **archived**  
  Folder containing archived attempts of prompt engineering and shadow calculation around the object.

- **validation_on_gt**  
  Contains precision and recall scores for pipeline optimization runs.

- **utils**  
  Folder contains modular helpers and wrappers for image processing, model interactions, and general-purpose utilities used across the project.

- **templates**  
  Contains prompts for GPT post-filtering pipeline.
