FROM nvcr.io/nvidia/pytorch:23.07-py3

# Add non-root user
ARG USERNAME='devHades'
RUN apt-get update \
    && apt-get install -y sudo \
    && useradd --create-home --shell /bin/bash ${USERNAME} \
    && echo "${USERNAME}:password" | chpasswd \
    && usermod -aG sudo ${USERNAME} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
WORKDIR /home/${USERNAME}


USER ${USERNAME}

CMD ["bash"]
