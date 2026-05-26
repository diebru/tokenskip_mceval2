# Derived McEval image with the language runtimes that the upstream
# multilingualnlp/mceval:latest is missing or has broken.
#
# Base image is Ubuntu 18.04. Several runtimes need third-party repos or
# tarball installs because 18.04 packages are too old or absent.
#
# Build:
#   cd tokenskip_mceval/docker && docker build -t mceval-full .
# Use:
#   IMAGE=mceval-full bash ../scripts/run_eval_docker.sh

FROM multilingualnlp/mceval:latest

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Remove the stale Dart GPG-broken apt source so apt-get update doesn't fail
RUN rm -f /etc/apt/sources.list.d/dart_stable.list \
          /etc/apt/sources.list.d/dart*.list \
          /etc/apt/sources.list.d/google-dart*.list 2>/dev/null || true

# 1. Base tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg wget xz-utils unzip \
        python3-bs4 \
    && rm -rf /var/lib/apt/lists/*

# 2. Node 16 via official tarball.
# Newer Node (18, 20) needs glibc 2.28+, Ubuntu 18.04 has 2.27. Node 16 is
# the last line that supports glibc 2.27, and is still recent enough for
# current TypeScript / CoffeeScript.
ENV NODE_VERSION=16.20.2
RUN curl -sSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
        -o /tmp/node.tar.xz \
    && mkdir -p /opt/node \
    && tar xJf /tmp/node.tar.xz -C /opt/node --strip-components=1 \
    && rm /tmp/node.tar.xz
ENV PATH="/opt/node/bin:${PATH}"

# 3. TypeScript + CoffeeScript via npm (global)
RUN npm install -g --silent typescript coffeescript

# 4. Apt-installable runtimes (versions that ship in 18.04 work as-is)
RUN apt-get update && apt-get install -y --no-install-recommends ghc && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends groovy && rm -rf /var/lib/apt/lists/*

# 4b. Use the rustup install that's already in the base image.
# The McEval image ships with rustup pre-installed at /root/.cargo/bin
# (a recent stable supporting edition 2021). Previously this was masked
# because apt's old rustc 1.41 at /usr/bin came earlier in PATH. Putting
# /root/.cargo/bin first makes the rustup toolchain the default.
# Re-running the rustup installer fails on Docker overlayfs with a
# cross-device link error, so we just lean on what's already there.
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustc --version && cargo --version

# 4c. Go 1.21 via official tarball.
# 18.04's apt golang-go is 1.10; McEval/data/go/go.mod requires go 1.18.
ENV GO_VERSION=1.21.5
RUN curl -sSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tar.gz \
    && tar xzf /tmp/go.tar.gz -C /opt \
    && rm /tmp/go.tar.gz
ENV PATH="/opt/go/bin:${PATH}"

# 4d. Scala 3 via official tarball.
# excute.py runs `scala -explain` which is a Scala 3-only flag. apt
# installs Scala 2 (which doesn't recognise -explain).
ENV SCALA3_VERSION=3.4.2
RUN curl -sSL "https://github.com/lampepfl/dotty/releases/download/${SCALA3_VERSION}/scala3-${SCALA3_VERSION}.tar.gz" \
        -o /tmp/scala3.tar.gz \
    && mkdir -p /opt/scala3 \
    && tar xzf /tmp/scala3.tar.gz -C /opt/scala3 --strip-components=1 \
    && rm /tmp/scala3.tar.gz
ENV PATH="/opt/scala3/bin:${PATH}"

# --- Third-party tarball installs ---

# Kotlin (official JetBrains release)
ENV KOTLIN_VERSION=1.9.24
RUN curl -sSL -o /tmp/kotlin.zip \
        "https://github.com/JetBrains/kotlin/releases/download/v${KOTLIN_VERSION}/kotlin-compiler-${KOTLIN_VERSION}.zip" \
    && unzip -q /tmp/kotlin.zip -d /opt \
    && rm /tmp/kotlin.zip
ENV PATH="/opt/kotlinc/bin:${PATH}"

# Julia
ENV JULIA_VERSION=1.10.4
RUN curl -sSL -o /tmp/julia.tar.gz \
        "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION%.*}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
    && mkdir -p /opt/julia \
    && tar xzf /tmp/julia.tar.gz -C /opt/julia --strip-components=1 \
    && rm /tmp/julia.tar.gz
ENV PATH="/opt/julia/bin:${PATH}"

# Dart 2.19 SDK tarball.
# The current Dart apt package needs glibc 2.31 (we have 2.27), and Dart 3.x
# raises the requirement further. Dart 2.19 is the last release line that
# works on Ubuntu 18.04 / glibc 2.27.
ENV DART_VERSION=2.19.6
RUN curl -sSL -o /tmp/dart.zip \
        "https://storage.googleapis.com/dart-archive/channels/stable/release/${DART_VERSION}/sdk/dartsdk-linux-x64-release.zip" \
    && unzip -q /tmp/dart.zip -d /opt \
    && rm /tmp/dart.zip
ENV PATH="/opt/dart-sdk/bin:${PATH}"

# Swift (Apple Linux tarball)
ENV SWIFT_VERSION=5.10.1
RUN curl -sSL -o /tmp/swift.tar.gz \
        "https://download.swift.org/swift-${SWIFT_VERSION}-release/ubuntu1804/swift-${SWIFT_VERSION}-RELEASE/swift-${SWIFT_VERSION}-RELEASE-ubuntu18.04.tar.gz" \
    && mkdir -p /opt/swift \
    && tar xzf /tmp/swift.tar.gz -C /opt/swift --strip-components=1 \
    && rm /tmp/swift.tar.gz \
    && apt-get update && apt-get install -y --no-install-recommends \
        libpython2.7 libncurses5 libtinfo5 libxml2 \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/swift/usr/bin:${PATH}"

# Smoke test: every required binary must be on PATH
RUN for bin in python3 perl pwsh ghc go rustc scala groovy node tsc coffee kotlinc julia dart swift; do \
        command -v "$bin" >/dev/null || { echo "missing: $bin"; exit 1; }; \
    done && echo "All language runtimes present."
