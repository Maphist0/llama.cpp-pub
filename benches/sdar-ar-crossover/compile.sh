#!/bin/sh
set -eu

: "${ANDROID_NDK:?Set ANDROID_NDK to your Android NDK directory}"
: "${OPENCL_DIR:?Set OPENCL_DIR to the directory containing include/ and lib/libOpenCL.so}"

src=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
ndk=$(CDPATH= cd -- "$ANDROID_NDK" && pwd)
opencl=$(CDPATH= cd -- "$OPENCL_DIR" && pwd)
build=${BUILD_DIR:-build-android-sdar}

if ! git -C "$src" apply --reverse --check "$src/benches/sdar-ar-crossover/measurement-only.patch" 2>/dev/null; then
    echo "Apply benches/sdar-ar-crossover/measurement-only.patch before compiling." >&2
    exit 1
fi

cmake -S "$src" -B "$build" \
    -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
    -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
    -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON \
    -DGGML_OPENCL_USE_ADRENO_KERNELS=ON -DGGML_OPENCL_PROFILING=OFF \
    -DOpenCL_INCLUDE_DIR="$opencl/include" \
    -DOpenCL_LIBRARY="$opencl/lib/libOpenCL.so" \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON

cmake --build "$build" --parallel "${JOBS:-8}" \
    --target llama-diffusion-cli llama-completion

printf 'Executables: %s/bin/llama-diffusion-cli and %s/bin/llama-completion\n' "$build" "$build"
