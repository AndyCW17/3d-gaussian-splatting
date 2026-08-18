"""
Gradio app: upload pre-run COLMAP output + the original photos, train a
Gaussian splat scene, download the result. No server framework, no job
queue -- one script, runs the pipeline synchronously while the user
watches a progress bar.

Run with:
    PYTHONPATH=src python3 app.py
Then open the printed local URL (usually http://127.0.0.1:7860).
"""

import os
import shutil
import tempfile
import gradio as gr
from splatproj.pipeline import run_pipeline

REQUIRED_COLMAP_FILES = {"cameras.txt", "images.txt", "points3D.txt"}


def _stage_job_folder(colmap_files, photo_files) -> tuple[str, str, str]:
    """
    Gradio hands us a list of file paths (strings) -- one per upload,
    already stored under Gradio's own temp directory with the ORIGINAL
    filename preserved as the basename. So we just need to copy each
    one into the right subfolder under that same basename; no renaming
    logic needed.

    (Verified this against the actual installed gradio version rather
    than assumed -- older Gradio versions returned file objects with a
    .name/.orig_name split instead of plain path strings, so this is
    worth re-checking if you ever upgrade gradio and this starts
    silently misbehaving.)

    Returns (job_dir, sparse_folder, images_folder).
    """
    job_dir = tempfile.mkdtemp(prefix="splat_job_")
    sparse_folder = os.path.join(job_dir, "sparse_txt")
    images_folder = os.path.join(job_dir, "images")
    os.makedirs(sparse_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    for f in colmap_files:
        shutil.copy(f, os.path.join(sparse_folder, os.path.basename(f)))

    for f in photo_files:
        shutil.copy(f, os.path.join(images_folder, os.path.basename(f)))

    return job_dir, sparse_folder, images_folder


def train_click(colmap_files, photo_files, num_iterations, progress=gr.Progress()):
    if not colmap_files or not photo_files:
        raise gr.Error("Upload both the COLMAP sparse files and the training photos first.")

    uploaded_names = {os.path.basename(f) for f in colmap_files}
    missing = REQUIRED_COLMAP_FILES - uploaded_names
    if missing:
        raise gr.Error(f"Missing required COLMAP file(s): {', '.join(sorted(missing))}")

    job_dir, sparse_folder, images_folder = _stage_job_folder(colmap_files, photo_files)
    output_dir = os.path.join(job_dir, "output")

    def on_iteration(iteration, loss_value, model):
        # gr.Progress wants a fraction in [0, 1] plus a short status string.
        # Same injected-callback slot train.py already exposes -- the
        # Gradio-specific part (progress bar) lives here, not in pipeline.py.
        if iteration % 20 == 0:
            progress(
                iteration / num_iterations,
                desc=f"iter {iteration}/{num_iterations} | loss {loss_value:.4f} | "
                     f"{len(model)} gaussians",
            )

    ply_path = run_pipeline(
        sparse_folder, images_folder, output_dir,
        num_iterations=int(num_iterations), device="cuda", on_iteration=on_iteration,
    )

    return ply_path


with gr.Blocks(title="3D Gaussian Splatting") as demo:
    gr.Markdown("# 3D Gaussian Splatting\nUpload COLMAP output + the original photos, train a scene.")

    with gr.Row():
        colmap_files = gr.File(
            file_count="multiple",
            label="COLMAP sparse output (cameras.txt, images.txt, points3D.txt)",
        )
        photo_files = gr.File(
            file_count="multiple",
            label="Training photos (same ones COLMAP was run on)",
        )

    num_iterations = gr.Slider(500, 5000, value=3000, step=100, label="Training iterations")
    train_button = gr.Button("Train", variant="primary")

    ply_output = gr.File(label="Trained scene (.ply)")

    train_button.click(
        fn=train_click,
        inputs=[colmap_files, photo_files, num_iterations],
        outputs=[ply_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)