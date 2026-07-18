# devel variant needed: gsplat compiles CUDA kernels at install time,
# which requires nvcc + headers, not just the CUDA runtime.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Avoid interactive prompts during apt installs (e.g. timezone selector)
ENV DEBIAN_FRONTEND=noninteractive

# Python itself isn't included in the CUDA base image
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first --- this is a deliberate ordering, not
# arbitrary. Docker caches each instruction as a "layer." If we copied
# all our source code before installing dependencies, changing a single
# line of Python would invalidate the cache and force a full dependency
# reinstall (minutes) on every rebuild. Copying requirements.txt alone
# means that expensive step only reruns when dependencies actually change.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual source code --- this layer changes often, but it's cheap.
COPY src/ src/

# A trivial sanity check for now. Day 7 replaces this with actually
# starting the FastAPI server.
CMD ["python3", "-c", "import torch; print('CUDA available:', torch.cuda.is_available())"]