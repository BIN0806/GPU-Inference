#!/usr/bin/env bash
# Minimal, pinned Linux compiler/runtime for CPU CI. No driver or GPU installation.
set -euo pipefail
if [[ $# != 1 || $(uname -s)-$(uname -m) != Linux-x86_64 ]]; then
  echo 'usage on Linux x86-64: install_cuda_compile_toolchain.sh /new/absolute/directory' >&2
  exit 2
fi
cuda_target=$1
case "$cuda_target" in /*) ;; *) echo 'destination must be absolute' >&2; exit 2 ;; esac
# A previous/partial toolchain must not silently contaminate a new build.
mkdir "$cuda_target"
download_dir=$(mktemp -d)
trap 'rm -rf "$download_dir"' EXIT
while read -r component version sha; do
  archive="$component-linux-x86_64-$version-archive.tar.xz"
  curl --fail --location --silent --show-error --retry 2 --max-time 180 \
    "https://developer.download.nvidia.com/compute/cuda/redist/$component/linux-x86_64/$archive" \
    -o "$download_dir/$archive"
  printf '%s  %s\n' "$sha" "$download_dir/$archive" | sha256sum --check --strict
  mkdir "$download_dir/$component"
  tar -xf "$download_dir/$archive" --strip-components=1 -C "$download_dir/$component"
  # Preserve component notices independently instead of overwriting each LICENSE.
  mkdir -p "$cuda_target/licenses/$component"
  for notice in "$download_dir/$component"/*LICENSE*; do
    if [[ -f "$notice" ]]; then cp "$notice" "$cuda_target/licenses/$component/"; fi
  done
  cp -a "$download_dir/$component/." "$cuda_target/"
done <<'PACKAGES'
cuda_nvcc 12.8.93 9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3
cuda_cudart 12.8.90 8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68
cuda_cccl 12.8.90 0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b
PACKAGES
# Redistributable archives use lib/; nvcc's default link search expects lib64/.
if [[ ! -e "$cuda_target/lib64" ]]; then ln -s lib "$cuda_target/lib64"; fi
"$cuda_target/bin/nvcc" --version
