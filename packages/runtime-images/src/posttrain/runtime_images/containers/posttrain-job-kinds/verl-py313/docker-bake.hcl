variable "CREATED" {
  default = ""
}

variable "DEPENDENCY_LOCK_SHA256" {
  default = ""
}

variable "FORK_REVISION" {
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

variable "RELEASE_CREATED" {
  default = ""
}

variable "RELEASE_SOURCE_REVISION" {
  default = ""
}

variable "RELEASE_VERSION" {
  default = "dev"
}

variable "SOURCE_REVISION" {
  default = ""
}

variable "SOURCE_REPOSITORY" {
  default = "https://github.com/carbonteq-ai/verl.git"
}

variable "VERSION" {
  default = "dev"
}

target "_common" {
  context = "."
  dockerfile = "containers/posttrain-job-kinds/verl-py313/Dockerfile"
  args = {
    CREATED = CREATED
    DEPENDENCY_LOCK_SHA256 = DEPENDENCY_LOCK_SHA256
    FORK_REVISION = FORK_REVISION
    LOCK_DIGEST = LOCK_DIGEST
    POSTTRAIN_BASE_IMAGE = POSTTRAIN_BASE_IMAGE
    RELEASE_CREATED = RELEASE_CREATED
    RELEASE_SOURCE_REVISION = RELEASE_SOURCE_REVISION
    RELEASE_VERSION = RELEASE_VERSION
    SOURCE_REVISION = SOURCE_REVISION
    SOURCE_REPOSITORY = SOURCE_REPOSITORY
    VERSION = VERSION
  }
}

target "posttrain-kind-online-rl-verl-py313" {
  inherits = ["_common"]
  target = "online-rl-verl-py313"
  tags = ["${REGISTRY}/posttrain-kind-online-rl-verl-py313:${RELEASE_VERSION}"]
  attest = [
    "type=provenance,mode=max",
    "type=sbom"
  ]
  output = [
    "type=image,push=true,compression=zstd,compression-level=3,force-compression=false,oci-mediatypes=true"
  ]
}

target "posttrain-kind-online-rl-verl-py313-smoke" {
  inherits = ["_common"]
  target = "online-rl-verl-py313-smoke"
  output = ["type=cacheonly"]
}
