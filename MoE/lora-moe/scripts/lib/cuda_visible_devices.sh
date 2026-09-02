#!/usr/bin/env bash

# Export physical GPU visibility before Python imports torch/DeepSpeed.
configure_cuda_visible_devices() {
  local visible_devices="$1"
  shift
  local cli_args=("$@")
  local i
  for ((i = 0; i < ${#cli_args[@]}; i++)); do
    case "${cli_args[$i]}" in
      --cuda-visible-devices)
        if ((i + 1 >= ${#cli_args[@]})); then
          echo "--cuda-visible-devices requires a value" >&2
          return 2
        fi
        visible_devices="${cli_args[$((i + 1))]}"
        ;;
      --cuda-visible-devices=*)
        visible_devices="${cli_args[$i]#*=}"
        ;;
    esac
  done
  export CUDA_VISIBLE_DEVICES="$visible_devices"
}
