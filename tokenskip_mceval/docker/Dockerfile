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

# 4. Apt-installable runtimes, one per layer so failures are obvious
RUN apt-get update && apt-get install -y --no-install-recommends ghc && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends golang-go && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends rustc cargo && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends scala && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends groovy && rm -rf /var/lib/apt/lists/*

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

# Dart (Google APT repo, freshly keyed)
RUN wget -qO- https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'wget -qO- https://storage.googleapis.com/download.dartlang.org/linux/debian/dart_stable.list \
             > /etc/apt/sources.list.d/dart_stable.list' \
    && apt-get update && apt-get install -y --no-install-recommends dart \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/usr/lib/dart/bin:${PATH}"

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
