variable "DEPENDENCY_LOCK_SHA256" {
  default = ""
}

variable "FORK_REVISION" {
  default = ""
}

variable "POSTTRAIN_BASE_IMAGE" {
  default = ""
}

variable "REGISTRY" {
  default = "registry.lan/carbonteq"
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
    DEPENDENCY_LOCK_SHA256 = DEPENDENCY_LOCK_SHA256
    FORK_REVISION = FORK_REVISION
    POSTTRAIN_BASE_IMAGE = POSTTRAIN_BASE_IMAGE
    SOURCE_REPOSITORY = SOURCE_REPOSITORY
  }
}

target "posttrain-kind-online-rl-verl-py313" {
  inherits = ["_common"]
  target = "online-rl-verl-py313"
  tags = ["${REGISTRY}/posttrain-kind-online-rl-verl-py313:${VERSION}"]
  attest = [
    "type=provenance,mode=max",
    "type=sbom"
  ]
  output = [
    "type=image,push=true,compression=zstd,compression-level=3,force-compression=true,oci-mediatypes=true"
  ]
}

target "posttrain-kind-online-rl-verl-py313-smoke" {
  inherits = ["_common"]
  target = "online-rl-verl-py313-smoke"
  output = ["type=cacheonly"]
}
