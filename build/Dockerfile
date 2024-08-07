ARG BASE=image-registry.openshift-image-registry.svc:5000/NAMESPACE/base:latest
ARG HIPBLASLT_IMAGE=image-registry.openshift-image-registry.svc:5000/NAMESPACE/export-hipblaslt:latest
ARG RCCL_IMAGE=image-registry.openshift-image-registry.svc:5000/NAMESPACE/export-rccl:latest
ARG FLASH_ATTN_IMAGE=image-registry.openshift-image-registry.svc:5000/NAMESPACE/export-flash-attn:latest
ARG TRITON_IMAGE=image-registry.openshift-image-registry.svc:5000/NAMESPACE/export-triton:latest
ARG AMDSMI_IMAGE=image-registry.openshift-image-registry.svc:5000/NAMESPACE/export-amdsmi:latest
ARG VLLM_IMAGE=image-registry.openshift-image-registry.svc:5000/NAMESPACE/export-vllm:latest
FROM ${HIPBLASLT_IMAGE} as export_hipblaslt
FROM ${RCCL_IMAGE} as export_rccl
FROM ${FLASH_ATTN_IMAGE} as export_flash_attn
FROM ${TRITON_IMAGE} as export_triton
FROM ${AMDSMI_IMAGE} as export_amdsmi
FROM ${VLLM_IMAGE} as export_vllm

FROM ${BASE}

RUN python3 -m pip install --upgrade pip && rm -rf /var/lib/apt/lists/*
# Error related to odd state for numpy 1.20.3 where there is no METADATA etc, but an extra LICENSES_bundled.txt.
# Manually remove it so that later steps of numpy upgrade can continue
RUN case "$(which python3)" in \
  *"/opt/conda/envs/py_3.9"*) \
  rm -rf /opt/conda/envs/py_3.9/lib/python3.9/site-packages/numpy-1.20.3.dist-info/;; \
  *) ;; esac

RUN --mount=type=bind,from=export_hipblaslt,src=/,target=/install \
  if ls /install/*.deb; then \
  apt-get purge -y hipblaslt \
  && dpkg -i /install/*.deb \
  && sed -i 's/, hipblaslt-dev \(.*\), hipcub-dev/, hipcub-dev/g' /var/lib/dpkg/status \
  && sed -i 's/, hipblaslt \(.*\), hipfft/, hipfft/g' /var/lib/dpkg/status; \
  fi

RUN --mount=type=bind,from=export_rccl,src=/,target=/install \
  if ls /install/*.deb; then \
  dpkg -i /install/*.deb \
  # RCCL needs to be installed twice
  && dpkg -i /install/*.deb \
  && sed -i 's/, rccl-dev \(.*\), rocalution/, rocalution/g' /var/lib/dpkg/status \
  && sed -i 's/, rccl \(.*\), rocalution/, rocalution/g' /var/lib/dpkg/status; \
  fi

RUN --mount=type=bind,from=export_flash_attn,src=/,target=/install \
  if ls /install/*.whl; then \
  # Preemptively uninstall to prevent pip same-version no-installs
  pip uninstall -y flash-attn \
  && pip install /install/*.whl; \
  fi

RUN --mount=type=bind,from=export_triton,src=/,target=/install \
  if ls /install/*.whl; then \
  # Preemptively uninstall to prevent pip same-version no-installs
  pip uninstall -y triton \
  && pip install /install/*.whl; \
  fi

RUN --mount=type=bind,from=export_amdsmi,src=/,target=/install \
  # Preemptively uninstall to prevent pip same-version no-installs
  pip uninstall -y amdsmi \
  && pip install /install/*.whl;

RUN python3 -m pip install --upgrade numba scipy huggingface-hub[cli]

# Install vLLM (and gradlib)
# Make sure punica kernels are built (for LoRA)
ENV VLLM_INSTALL_PUNICA_KERNELS=1
RUN --mount=type=bind,from=export_vllm,src=/,target=/install \
  cd /install \
  && pip install -U -r requirements-rocm.txt \
  && case "$(ls /opt | grep -Po 'rocm-[0-9]\.[0-9]')" in \
  *"rocm-6.0"*) \
  patch /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h rocm_patch/rocm_bf16.patch;; \
  *"rocm-6.1"*) \
  cp rocm_patch/libamdhip64.so.6 /opt/rocm/lib/libamdhip64.so.6;; \
  *) ;; esac \
  && pip uninstall -y vllm gradlib \
  && pip install *.whl

# Copy over the benchmark scripts as well
COPY --from=export_vllm /benchmarks /app/vllm/benchmarks
COPY --from=export_vllm /tests /app/vllm/tests
COPY --from=export_vllm /examples /app/vllm/examples
COPY --from=export_vllm /.buildkite /app/vllm/.buildkite


ENV RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
ENV TOKENIZERS_PARALLELISM=false

# Performance environment variable.
ENV HIP_FORCE_DEV_KERNARG=1

CMD ["/bin/bash"]
