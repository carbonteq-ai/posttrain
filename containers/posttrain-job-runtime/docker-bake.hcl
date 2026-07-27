variable "BASE_IMAGE" {
  default = ""
}

variable "SOURCE_DIGEST" {
  default = ""
}

variable "LOCK_DIGEST" {
  default = ""
}

variable "RUNTIME_PROFILE" {
  default = "framework/job@1"
}

group "default" {
  targets = ["posttrain-job-runtime"]
}

target "posttrain-job-runtime" {
  context = "."
  dockerfile = "containers/posttrain-job-runtime/Dockerfile"
  output = [
    "type=image,push=true,compression=zstd,compression-level=3,force-compression=true,oci-mediatypes=true"
  ]
  attest = [
    "type=provenance,mode=max",
    "type=sbom"
  ]
  args = {
    BASE_IMAGE = BASE_IMAGE
    SOURCE_DIGEST = SOURCE_DIGEST
    LOCK_DIGEST = LOCK_DIGEST
    RUNTIME_PROFILE = RUNTIME_PROFILE
  }
}
