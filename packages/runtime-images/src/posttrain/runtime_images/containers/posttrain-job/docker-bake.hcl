variable "FRAMEWORK_SOURCE_DIGEST" {
  default = ""
}

variable "ALLOW_DEFERRED_QUALIFICATION" {
  default = "0"
}

variable "CODE_REQUIREMENTS_DIGEST" {
  default = ""
}

variable "IMAGE_REPOSITORY" {
  default = "registry.lan/carbonteq/posttrain-job"
}

variable "IMAGE_TAG" {
  default = "dev"
}

variable "PACKAGE_KEY" {
  default = ""
}

variable "PYTHON_INDEX_URL" {
  default = ""
}

variable "PROJECT_CONFIG_DIGEST" {
  default = ""
}

variable "JOB_KIND" {
  default = ""
}

variable "POSTTRAIN_KIND_IMAGE" {
  default = ""
}

variable "PROJECT_SOURCE_DIGEST" {
  default = ""
}

variable "RESOLVED_INPUTS_DIGEST" {
  default = ""
}

variable "RESOLVED_CONFIG_DIGEST" {
  default = ""
}

variable "RUNTIME_DEPENDENCIES_DIGEST" {
  default = ""
}

variable "RUNTIME_VARIANT" {
  default = ""
}

variable "STAGED_CONTEXT" {
  default = "containers/posttrain-job/fixtures/minimal-context"
}

group "default" {
  targets = ["posttrain-job"]
}

group "smoke" {
  targets = ["posttrain-job-smoke"]
}

target "_common" {
  context = "containers/posttrain-job"
  contexts = {
    job-context = STAGED_CONTEXT
  }
  dockerfile = "Dockerfile"
  args = {
    ALLOW_DEFERRED_QUALIFICATION = ALLOW_DEFERRED_QUALIFICATION
    CODE_REQUIREMENTS_DIGEST = CODE_REQUIREMENTS_DIGEST
    FRAMEWORK_SOURCE_DIGEST = FRAMEWORK_SOURCE_DIGEST
    JOB_KIND = JOB_KIND
    PACKAGE_KEY = PACKAGE_KEY
    POSTTRAIN_KIND_IMAGE = POSTTRAIN_KIND_IMAGE
    PYTHON_INDEX_URL = PYTHON_INDEX_URL
    PROJECT_CONFIG_DIGEST = PROJECT_CONFIG_DIGEST
    PROJECT_SOURCE_DIGEST = PROJECT_SOURCE_DIGEST
    RESOLVED_INPUTS_DIGEST = RESOLVED_INPUTS_DIGEST
    RESOLVED_CONFIG_DIGEST = RESOLVED_CONFIG_DIGEST
    RUNTIME_DEPENDENCIES_DIGEST = RUNTIME_DEPENDENCIES_DIGEST
    RUNTIME_VARIANT = RUNTIME_VARIANT
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

target "posttrain-job" {
  inherits = ["_published"]
  target = "runtime"
  tags = ["${IMAGE_REPOSITORY}:${IMAGE_TAG}"]
}

target "posttrain-job-smoke" {
  inherits = ["_common"]
  target = "smoke"
  output = ["type=cacheonly"]
}
