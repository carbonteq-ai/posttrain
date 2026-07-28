variable "CREATED" {
  default = ""
}

variable "LOCK_DIGEST" {
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
  targets = ["posttrain-base"]
}

group "smoke" {
  targets = ["posttrain-base-smoke"]
}

target "_metadata" {
  args = {
    CREATED = CREATED
    LOCK_DIGEST = LOCK_DIGEST
    SOURCE_REVISION = SOURCE_REVISION
    VERSION = VERSION
  }
}

target "_published" {
  inherits = ["_metadata"]
  context = "."
  dockerfile = "containers/posttrain-base/Dockerfile"
  output = [
    "type=image,push=true,compression=zstd,compression-level=1,force-compression=false,oci-mediatypes=true"
  ]
}

target "posttrain-base" {
  inherits = ["_published"]
  target = "runtime"
  tags = ["${REGISTRY}/posttrain-base:${VERSION}"]
}

target "posttrain-base-smoke" {
  inherits = ["_metadata"]
  context = "."
  dockerfile = "containers/posttrain-base/Dockerfile"
  target = "smoke"
  output = ["type=cacheonly"]
}
