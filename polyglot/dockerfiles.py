# IF you change the base image, you need to rebuild all images (run with --force_rebuild)
_DOCKERFILE_BASE = r"""
FROM buildpack-deps:jammy

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    cmake \
    libboost-all-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ca-certificates-java \
    openjdk-21-jdk \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Go with architecture detection
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        GOARCH="amd64"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        GOARCH="arm64"; \
    else \
        false; \
    fi && \
    curl -L "https://golang.org/dl/go1.21.5.linux-$GOARCH.tar.gz" -o go.tar.gz && \
    tar -C /usr/local -xzf go.tar.gz && \
    rm go.tar.gz
ENV PATH="/usr/local/go/bin:${{PATH}}"

# Tool homes are fixed because Apptainer runs tests with HOME set to the caller's
# home, not /root. rustup would find no toolchain there, and that home is a small
# tmpfs that the Gradle download does not fit in. Under Docker these are the defaults.
ENV RUSTUP_HOME=/root/.rustup
ENV CARGO_HOME=/root/.cargo
ENV GRADLE_USER_HOME=/root/.gradle

# Install Rust
ADD https://sh.rustup.rs /tmp/rustup.sh
# Minimal profile: every sandbox unpacks this image onto a shared temp disk, and the
# default profile's HTML docs alone are about 0.9 GB. cargo and rustc are all tests need.
RUN chmod +x /tmp/rustup.sh && /tmp/rustup.sh -y --profile minimal && rm /tmp/rustup.sh
ENV PATH="/root/.cargo/bin:${{PATH}}"

# Bake in what the Java problems' Gradle wrapper fetches on first use. Otherwise every
# sandbox downloads it inside the 120 s eval limit, and no Java test ever runs. The
# directory name is the wrapper's hash of the distribution URL in the problems'
# gradle-wrapper.properties. The seed build resolves the problems' dependencies.
RUN mkdir -p /root/.gradle/wrapper/dists/gradle-8.7-bin/bhs2wmbdwecv87pi65oeuq5iu && \
    cd /root/.gradle/wrapper/dists/gradle-8.7-bin/bhs2wmbdwecv87pi65oeuq5iu && \
    curl -fsSL https://services.gradle.org/distributions/gradle-8.7-bin.zip -o gradle-8.7-bin.zip && \
    unzip -q gradle-8.7-bin.zip && \
    rm gradle-8.7-bin.zip && \
    touch gradle-8.7-bin.zip.ok && \
    mkdir -p /tmp/gradle-seed/src/test/java && \
    cd /tmp/gradle-seed && \
    printf '%s\n' \
        'plugins {{ id "java" }}' \
        'repositories {{ mavenCentral() }}' \
        'dependencies {{' \
        '    implementation "org.json:json:20190722"' \
        '    implementation "io.reactivex.rxjava2:rxjava:2.2.12"' \
        '    testImplementation platform("org.junit:junit-bom:5.10.0")' \
        '    testImplementation "org.junit.jupiter:junit-jupiter"' \
        '    testImplementation "org.assertj:assertj-core:3.25.1"' \
        '}}' \
        'test {{ useJUnitPlatform() }}' > build.gradle && \
    printf '%s\n' \
        'import org.junit.jupiter.api.Test;' \
        'class SeedTest {{ @Test void seed() {{ }} }}' > src/test/java/SeedTest.java && \
    /root/.gradle/wrapper/dists/gradle-8.7-bin/bhs2wmbdwecv87pi65oeuq5iu/gradle-8.7/bin/gradle --no-daemon test && \
    cd / && \
    rm -rf /tmp/gradle-seed

# Install Node.js and dependencies
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /npm-install && \
    cd /npm-install && \
    npm init -y && \
    npm install \
    jest \
    @babel/core@7.25.2 \
    @exercism/babel-preset-javascript@0.2.1 \
    @exercism/eslint-config-javascript@0.6.0 \
    @types/jest@29.5.12 \
    @types/node@20.12.12 \
    babel-jest@29.6.4 \
    core-js@3.37.1 \
    eslint@8.49.0

# Download and install conda
RUN wget 'https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-{conda_arch}.sh' -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/miniconda3
# Add conda to PATH
ENV PATH=/opt/miniconda3/bin:$PATH
# Add conda to shell startup scripts like .bashrc (DO NOT REMOVE THIS)
RUN conda init --all
RUN conda config --append channels conda-forge

RUN adduser --disabled-password --gecos 'dog' nonroot
"""

_DOCKERFILE_ENV = r"""FROM --platform={platform} pb.base.{arch}:latest

COPY ./setup_env.sh /root/
RUN chmod +x /root/setup_env.sh
RUN /bin/bash -c "source ~/.bashrc && /root/setup_env.sh"
# The package cache is about 1.3 GB that every sandbox would unpack and never read.
RUN conda clean -afy

WORKDIR /testbed/

# Automatically activate the testbed environment
RUN echo "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed" > /root/.bashrc
"""

_DOCKERFILE_INSTANCE = r"""FROM --platform={platform} {env_image_name}

COPY ./setup_repo.sh /root/
COPY ./polyglot /polyglot
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
"""


def get_dockerfile_base(platform, arch):
    if arch == "arm64":
        conda_arch = "aarch64"
    else:
        conda_arch = arch
    return _DOCKERFILE_BASE.format(conda_arch=conda_arch)


def get_dockerfile_env(platform, arch):
    return _DOCKERFILE_ENV.format(platform=platform, arch=arch)


def get_dockerfile_instance(platform, env_image_name):
    return _DOCKERFILE_INSTANCE.format(platform=platform, env_image_name=env_image_name)
