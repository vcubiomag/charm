#!/usr/bin/env bash
# Installs the native dependencies of a wheel build into a manylinux_2_34 container.
set -euo pipefail

dnf install -y gmp-devel mpfr-devel
command -v ninja >/dev/null || pipx install ninja

work=$(mktemp -d)

# AlmaLinux 9 ships Boost 1.75, whose Multiprecision GMP backend doesn't compile with GCC: it converts gmp_int to
# __float128 through lexical_cast, which can't stream that type. Fixed in Boost 1.79. CGAL only needs the headers,
# and the version matches the Homebrew bottle the macOS wheels are built against.
boost_version=1.92.0
boost_sha256=ea7b982002cc9dfbe59b0b217b206f470dc75f3de0bb2973d844118934d82411
curl -fsSL -o "$work/boost.tar.xz" \
  "https://github.com/boostorg/boost/releases/download/boost-${boost_version}/boost-${boost_version}-b2-nodocs.tar.xz"
echo "${boost_sha256}  $work/boost.tar.xz" | sha256sum -c -
tar -xJf "$work/boost.tar.xz" -C "$work"
(cd "$work/boost-${boost_version}" && ./bootstrap.sh && ./b2 -d0 install --with-headers --prefix=/usr/local)

# AlmaLinux 9 ships TBB 2020, which predates oneTBB. Install into /usr so auditwheel finds it without ldconfig changes.
tbb_version=2023.1.0
tbb_sha256=191288b52e1e6b17198000b64d77d194bb65e791be46ebc606e9b091781e2070
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
