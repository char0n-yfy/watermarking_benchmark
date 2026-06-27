CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE run_status AS ENUM (
  'draft',
  'queued',
  'running',
  'succeeded',
  'failed',
  'cancelled',
  'partially_failed'
);

CREATE TYPE package_status AS ENUM (
  'uploaded',
  'reviewed',
  'built',
  'enabled',
  'rejected'
);

CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT NOT NULL UNIQUE,
  display_name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE datasets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id UUID REFERENCES users(id),
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE dataset_versions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  version TEXT NOT NULL,
  storage_uri TEXT NOT NULL,
  checksum TEXT NOT NULL,
  sample_count INTEGER NOT NULL CHECK (sample_count >= 0),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (dataset_id, version),
  UNIQUE (checksum)
);

CREATE TABLE samples (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset_version_id UUID NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
  sample_key TEXT NOT NULL,
  storage_uri TEXT NOT NULL,
  checksum TEXT NOT NULL,
  width INTEGER,
  height INTEGER,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE (dataset_version_id, sample_key)
);

CREATE TABLE algorithm_packages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id UUID REFERENCES users(id),
  name TEXT NOT NULL,
  status package_status NOT NULL DEFAULT 'uploaded',
  source_artifact_id UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE algorithm_versions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  package_id UUID NOT NULL REFERENCES algorithm_packages(id) ON DELETE CASCADE,
  version TEXT NOT NULL,
  image_ref TEXT,
  entrypoint TEXT NOT NULL DEFAULT '',
  requires_gpu BOOLEAN NOT NULL DEFAULT false,
  config_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
  status package_status NOT NULL DEFAULT 'uploaded',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (package_id, version)
);

CREATE TABLE model_artifacts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  algorithm_version_id UUID REFERENCES algorithm_versions(id),
  name TEXT NOT NULL,
  storage_uri TEXT NOT NULL,
  checksum TEXT NOT NULL UNIQUE,
  size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
  mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE attack_methods (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL UNIQUE,
  family TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  config_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
  enabled BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE attack_presets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  method_id UUID NOT NULL REFERENCES attack_methods(id),
  name TEXT NOT NULL,
  params JSONB NOT NULL DEFAULT '{}'::jsonb,
  strengths JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (method_id, name)
);

CREATE TABLE experiment_specs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id UUID REFERENCES users(id),
  name TEXT NOT NULL,
  spec JSONB NOT NULL,
  status run_status NOT NULL DEFAULT 'draft',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE experiment_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  spec_id UUID NOT NULL REFERENCES experiment_specs(id),
  status run_status NOT NULL DEFAULT 'queued',
  artifact_root TEXT NOT NULL,
  cell_count INTEGER NOT NULL CHECK (cell_count >= 0),
  progress INTEGER NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
  log_path TEXT,
  worker_id TEXT,
  cancel_requested BOOLEAN NOT NULL DEFAULT false,
  error TEXT,
  queued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
);

CREATE TABLE experiment_cells (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
  cell_key TEXT NOT NULL,
  dataset_version_id UUID NOT NULL REFERENCES dataset_versions(id),
  algorithm_version_id UUID NOT NULL REFERENCES algorithm_versions(id),
  attack_preset_id UUID NOT NULL REFERENCES attack_presets(id),
  seed INTEGER NOT NULL,
  status run_status NOT NULL DEFAULT 'queued',
  sample_count INTEGER NOT NULL DEFAULT 0 CHECK (sample_count >= 0),
  bit_accuracy DOUBLE PRECISION,
  bit_error_rate DOUBLE PRECISION,
  elapsed_ms DOUBLE PRECISION,
  output_uri TEXT,
  manifest_uri TEXT,
  error TEXT,
  summary JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (run_id, cell_key)
);

CREATE TABLE worker_heartbeats (
  worker_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  pid INTEGER NOT NULL,
  device TEXT NOT NULL,
  current_run_id UUID REFERENCES experiment_runs(id),
  message TEXT,
  last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE artifacts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  artifact_type TEXT NOT NULL,
  storage_uri TEXT NOT NULL UNIQUE,
  checksum TEXT NOT NULL,
  size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
  mime_type TEXT NOT NULL,
  owner_entity TEXT NOT NULL,
  owner_id UUID NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (checksum, owner_entity, owner_id)
);

CREATE TABLE metric_summaries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  cell_id UUID NOT NULL REFERENCES experiment_cells(id) ON DELETE CASCADE,
  metric_name TEXT NOT NULL,
  metric_value DOUBLE PRECISION NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (cell_id, metric_name)
);

CREATE TABLE sandbox_builds (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  algorithm_version_id UUID NOT NULL REFERENCES algorithm_versions(id) ON DELETE CASCADE,
  status package_status NOT NULL DEFAULT 'uploaded',
  image_ref TEXT,
  logs_uri TEXT,
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
