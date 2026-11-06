#!/usr/bin/env bash
# Installs the native dependencies of a wheel build into a manylinux_2_34 container.
set -euo pipefail

dnf install -y boost-devel gmp-devel mpfr-devel
command -v ninja >/dev/null || pipx install ninja

# AlmaLinux 9 ships TBB 2020, which predates oneTBB. Install into /usr so auditwheel finds it without ldconfig changes.
tbb_version=2023.1.0
tbb_sha256=191288b52e1e6b17198000b64d77d194bb65e791be46ebc606e9b091781e2070
work=$(mktemp -d)
curl -fsSL -o "$work/tbb.tar.gz" "https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v${tbb_version}.tar.gz"
echo "${tbb_sha256}  $work/tbb.tar.gz" | sha256sum -c -
tar -xzf "$work/tbb.tar.gz" -C "$work"
cmake -S "$work/oneTBB-${tbb_version}" -B "$work/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr \
  -DTBB_TEST=OFF \
  -DTBB_EXAMPLES=OFF \
  -DTBB_DISABLE_HWLOC_AUTOMATIC_SEARCH=ON
cmake --build "$work/build" --target install
rm -rf "$work"
