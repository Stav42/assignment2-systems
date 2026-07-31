import subprocess
import modal

image = modal.Image.from_registry("nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04", add_python="3.12").uv_sync(extra_options="--no-install-package cs336-basics").apt_install("libdw1")

image = image.add_local_python_source("cs336_basics", "cs336_systems")
app = modal.App("cs336_systems", image=image)

@app.function(gpu="A100", image=image)
def invoke():
    report_data = None
    subprocess.run(["nsys", "profile", "-o", "benchmark", "--", "python", "cs336_systems/base_bench.py", "--device", "cuda", "--forward_only"])
    with open("/root/benchmark.nsys-rep", "rb") as f:
        report_data = f.read()

    return report_data


if __name__ == "__main__":
    # Run the benchmark script using subprocess
    with modal.enable_output():
        with app.run():
            data = invoke.remote()

    with open("benchmark.nsys-rep", "wb") as f:
        f.write(data)

            

