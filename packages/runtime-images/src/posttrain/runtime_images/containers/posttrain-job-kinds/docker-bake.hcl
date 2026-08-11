variable "CREATED" {
  default = ""
}

variable "LOCK_DIGEST" {
  default = ""
}

variable "POSTTRAIN_BASE_IMAGE" {
  default = ""
}

variable "REGISTRY" {
  default = "registry.lan/carbonteq"
}

variable "SOURCE_REVISION" {
  default = ""
}

variable "VERSION" {
  default = "dev"
}

group "default" {
  targets = [
    "posttrain-kind-supervised",
    "posttrain-kind-online-rl-trl-py312",
    "posttrain-kind-online-rl-verl-py313",
    "posttrain-kind-eval",
    "posttrain-kind-serve",
    "posttrain-kind-transform"
  ]
}

group "smoke" {
  targets = [
    "posttrain-kind-supervised-smoke",
    "posttrain-kind-online-rl-trl-py312-smoke",
    "posttrain-kind-online-rl-verl-py313-smoke",
    "posttrain-kind-eval-smoke",
    "posttrain-kind-serve-smoke",
    "posttrain-kind-transform-smoke"
  ]
}

target "_common" {
  context = "."
  dockerfile = "containers/posttrain-job-kinds/Dockerfile"
  args = {
    CREATED = CREATED
    LOCK_DIGEST = LOCK_DIGEST
    POSTTRAIN_BASE_IMAGE = POSTTRAIN_BASE_IMAGE
    SOURCE_REVISION = SOURCE_REVISION
    VERSION = VERSION
  }
}

target "_published" {
  inherits = ["_common"]
  output = [
    "type=image,push=true,compression=zstd,compression-level=1,force-compression=false,oci-mediatypes=true"
  ]
}

target "_smoke" {
  inherits = ["_common"]
  output = ["type=cacheonly"]
}

target "posttrain-kind-supervised" {
  inherits = ["_published"]
  target = "supervised"
  tags = ["${REGISTRY}/posttrain-kind-supervised:${VERSION}"]
}

target "posttrain-kind-online-rl-trl-py312" {
  inherits = ["_published"]
  target = "online-rl-trl-py312"
  tags = ["${REGISTRY}/posttrain-kind-online-rl-trl-py312:${VERSION}"]
}

target "posttrain-kind-online-rl-verl-py313" {
  context = "."
  dockerfile = "containers/posttrain-job-kinds/verl-py313/Dockerfile"
  target = "online-rl-verl-py313"
  args = {
    CREATED = CREATED
    DEPENDENCY_LOCK_SHA256 = "df7769f306ef8606fed37fd89c3fa2589ac72f39a45e4743816368a1acffe69f"
    FORK_REVISION = "a6fe39c22719ec981ed8544ad8feffd59995cc13"
    LOCK_DIGEST = LOCK_DIGEST
    POSTTRAIN_BASE_IMAGE = POSTTRAIN_BASE_IMAGE
    SOURCE_REVISION = SOURCE_REVISION
    SOURCE_REPOSITORY = "https://github.com/carbonteq-ai/verl.git"
    VERSION = VERSION
  }
  tags = ["${REGISTRY}/posttrain-kind-online-rl-verl-py313:${VERSION}"]
  output = [
    "type=image,push=true,compression=zstd,compression-level=1,force-compression=false,oci-mediatypes=true"
  ]
}

target "posttrain-kind-eval" {
  inherits = ["_published"]
  target = "eval"
  tags = ["${REGISTRY}/posttrain-kind-eval:${VERSION}"]
}

target "posttrain-kind-serve" {
  inherits = ["_published"]
  target = "serve"
  tags = ["${REGISTRY}/posttrain-kind-serve:${VERSION}"]
}

target "posttrain-kind-transform" {
  inherits = ["_published"]
  target = "transform"
  tags = ["${REGISTRY}/posttrain-kind-transform:${VERSION}"]
}

target "posttrain-kind-supervised-smoke" {
  inherits = ["_smoke"]
  target = "supervised-smoke"
}

target "posttrain-kind-online-rl-trl-py312-smoke" {
  inherits = ["_smoke"]
  target = "online-rl-trl-py312-smoke"
}

target "posttrain-kind-online-rl-verl-py313-smoke" {
  context = "."
  dockerfile = "containers/posttrain-job-kinds/verl-py313/Dockerfile"
  target = "online-rl-verl-py313-smoke"
  args = {
    CREATED = CREATED
    DEPENDENCY_LOCK_SHA256 = "df7769f306ef8606fed37fd89c3fa2589ac72f39a45e4743816368a1acffe69f"
    FORK_REVISION = "a6fe39c22719ec981ed8544ad8feffd59995cc13"
    LOCK_DIGEST = LOCK_DIGEST
    POSTTRAIN_BASE_IMAGE = POSTTRAIN_BASE_IMAGE
    SOURCE_REVISION = SOURCE_REVISION
    SOURCE_REPOSITORY = "https://github.com/carbonteq-ai/verl.git"
    VERSION = VERSION
  }
  output = ["type=cacheonly"]
}

target "posttrain-kind-eval-smoke" {
  inherits = ["_smoke"]
  target = "eval-smoke"
}

target "posttrain-kind-serve-smoke" {
  inherits = ["_smoke"]
  target = "serve-smoke"
}

target "posttrain-kind-transform-smoke" {
  inherits = ["_smoke"]
  target = "transform-smoke"
}
