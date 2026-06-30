#!/usr/bin/env bash

autodl_node_major() {
  if ! command -v node >/dev/null 2>&1; then
    echo "0"
    return 0
  fi
  node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo "0"
}

autodl_ensure_node() {
  local required_major="${WM_BENCH_NODE_VERSION:-20}"
  local current_major
  current_major="$(autodl_node_major)"

  if [[ "${current_major}" =~ ^[0-9]+$ ]] && ((current_major >= required_major)); then
    return 0
  fi

  if [[ "${WM_BENCH_AUTO_INSTALL_NODE:-1}" == "0" ]]; then
    echo "Node.js ${required_major}+ is required but was not found." >&2
    echo "Install Node.js ${required_major}+ or set WM_BENCH_AUTO_INSTALL_NODE=1." >&2
    return 1
  fi

  if command -v conda >/dev/null 2>&1; then
    echo "Installing Node.js ${required_major}.x with conda..."
    conda install -y -c conda-forge "nodejs>=${required_major},<$((${required_major} + 1))"
    hash -r
    current_major="$(autodl_node_major)"
    if [[ "${current_major}" =~ ^[0-9]+$ ]] && ((current_major >= required_major)); then
      return 0
    fi
  fi

  echo "Unable to prepare Node.js ${required_major}+ automatically." >&2
  echo "Install Node.js ${required_major}+ and rerun: bash infra/autodl/start.sh" >&2
  return 1
}

autodl_ensure_pnpm() {
  local pnpm_version="${WM_BENCH_PNPM_VERSION:-9.15.0}"

  if command -v pnpm >/dev/null 2>&1; then
    return 0
  fi

  if command -v corepack >/dev/null 2>&1; then
    if corepack enable && corepack prepare "pnpm@${pnpm_version}" --activate; then
      hash -r
      if command -v pnpm >/dev/null 2>&1; then
        return 0
      fi
      if corepack pnpm --version >/dev/null 2>&1; then
        return 0
      fi
    fi
  fi

  if [[ "${WM_BENCH_AUTO_INSTALL_NODE:-1}" != "0" ]] && command -v npm >/dev/null 2>&1; then
    echo "Installing pnpm ${pnpm_version} with npm..."
    npm install -g "pnpm@${pnpm_version}"
    hash -r
    command -v pnpm >/dev/null 2>&1 && return 0
  fi

  echo "pnpm ${pnpm_version} is required but was not found." >&2
  echo "Install pnpm or keep corepack available, then rerun the AutoDL startup script." >&2
  return 1
}

autodl_pnpm() {
  if command -v pnpm >/dev/null 2>&1; then
    pnpm "$@"
  else
    corepack pnpm "$@"
  fi
}
