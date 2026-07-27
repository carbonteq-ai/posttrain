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
    "posttrain-kind-eval",
    "posttrain-kind-serve",
    "posttrain-kind-transform"
  ]
}

group "smoke" {
  targets = [
    "posttrain-kind-supervised-smoke",
    "posttrain-kind-online-rl-trl-py312-smoke",
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
  attest = [
    "type=provenance,mode=max",
    "type=sbom"
  ]
  output = [
    "type=image,push=true,compression=zstd,compression-level=3,force-compression=true,oci-mediatypes=true"
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
