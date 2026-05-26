# Derived McEval image with the language runtimes that the upstream
# multilingualnlp/mceval:latest is missing or has broken.
#
# Base image is Ubuntu 18.04. Many runtimes need third-party repos or
# direct tarball installs because Ubuntu 18.04's package versions are
# too old (e.g. Julia, Swift) or not in the default repos (Kotlin, Dart).
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

# Base tooling + system-package runtimes that ARE in 18.04 apt repos
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        curl ca-certificates gnupg wget xz-utils unzip \
        # Already-broken in image:
        python3-bs4 \
        # Easy via apt:
        ghc \
        golang-go \
        rustc cargo \
        scala \
        groovy \
        nodejs npm \
        && rm -rf /var/lib/apt/lists/*

# TypeScript + CoffeeScript via npm (global)
RUN npm install -g --silent typescript coffeescript

# --- Languages that need third-party installs ---

# Kotlin (sdkman is heavy; just grab the official kotlinc release)
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

# Dart (use the Google APT repo properly this time)
RUN wget -qO- https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'wget -qO- https://storage.googleapis.com/download.dartlang.org/linux/debian/dart_stable.list \
             > /etc/apt/sources.list.d/dart_stable.list' \
    && apt-get update -qq && apt-get install -y -qq --no-install-recommends dart \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/usr/lib/dart/bin:${PATH}"

# Swift (Linux toolchain, official Apple tarball)
ENV SWIFT_VERSION=5.10.1
RUN curl -sSL -o /tmp/swift.tar.gz \
        "https://download.swift.org/swift-${SWIFT_VERSION}-release/ubuntu1804/swift-${SWIFT_VERSION}-RELEASE/swift-${SWIFT_VERSION}-RELEASE-ubuntu18.04.tar.gz" \
    && mkdir -p /opt/swift \
    && tar xzf /tmp/swift.tar.gz -C /opt/swift --strip-components=1 \
    && rm /tmp/swift.tar.gz \
    && apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        libpython2.7 libncurses5 libtinfo5 libxml2 \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/swift/usr/bin:${PATH}"

# Smoke test: every required binary on PATH
RUN for bin in python3 perl pwsh ghc go rustc scala groovy node tsc coffee kotlinc julia dart swift; do \
        command -v "$bin" >/dev/null || { echo "missing: $bin"; exit 1; }; \
    done && echo "All language runtimes present."
