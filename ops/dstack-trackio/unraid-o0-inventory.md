# Unraid services-host inventory

Status: O0 read-only inventory captured on 2026-07-26. No container, virtual
machine, firewall, DHCP, DNS, or storage configuration was changed.

This document records the observed deployment surface for the internal
post-training services. It contains no credentials. Re-run the inventory before
O1 because addresses, free space, and running guests can change.

## Observed host

| Item | Observation | Consequence |
| --- | --- | --- |
| Host | `dev-v2`, Unraid 7.1.4, `192.168.110.58` | Keep the Unraid management host separate from application ingress |
| Compute | Intel Xeon Platinum 8160M, 96 logical CPUs, about 1 TiB RAM | Enough CPU and RAM for the control services and a non-HA Doris deployment |
| Array | Five mounted 3.7 TB XFS data disks, about 19 TB aggregate | Backup and cold-retention target, not the primary analytics write path |
| NVMe pool | Lexar NQ780 2 TB XFS at `/mnt/nvme-pool`, about 1.4 TB free | Primary service database and hot-data tier |
| Docker | Docker 27.5.1; only the Cloudflare tunnel container was running | No conflicting application stack was found |
| Docker storage | 20 GB loop-backed Docker image under `/mnt/user/system` | Too small for the proposed stack; do not deploy Doris or registry data into it |
| Host ingress | Unraid owns TCP 80 and 443 on `.58` | Do not put application virtual hosts directly on the Unraid address |
| Docker LAN | `br0` is a macvlan on `192.168.110.0/24` | Direct container LAN addresses are possible but add lifecycle and collision risk |

The supplied access file is local-only, ignored by Git, and restricted to mode
`0600`. It must never be copied into compose files, shell history, execution
logs, Trackio, or repository artifacts.

## Existing virtualization surface

The Unraid host already runs these guests:

| Guest | Observed allocation | Relevant observation |
| --- | --- | --- |
| `Dokploy` | 12 vCPU, about 32 GiB RAM, 500 GB raw NVMe disk | Active at `192.168.110.61`; ports 22, 80, 443, and 3000 respond |
| `Fleetdm` | 4 vCPU, 8 GiB RAM | Not part of this deployment |
| `DaytonaSandbox` | 6 vCPU, 8 GiB RAM | Not part of this deployment |
| `Ct-PasswordManager` | 2 vCPU, 4 GiB RAM | Not part of this deployment |

The Dokploy guest already provides Traefik-backed HTTP ingress, but it is
explicitly excluded from the post-training service dependency graph. Existing
Dokploy workloads, upgrades, or failure must not control dstack, Trackio,
Doris, registry access, or Observatory availability.

## UniFi and local DNS

Observed network state:

- UniFi Network 10.4.57 on UniFi OS 5.1.19;
- VLAN 110, `Internal Services`, uses `192.168.110.0/24`;
- the DHCP pool is `192.168.110.6` through `192.168.110.254`;
- `192.168.110.61` is leased to the unrelated Dokploy VM and has no fixed-IP
  reservation;
- existing local records use the `.lan` suffix;
- `dev-v2-server.lan` already resolves to the Unraid host at `.58`.

The immediate work is not a DNS software change. The two initial service VMs need
distinct, fixed addresses before records are created. Do not point any
post-training service name at Dokploy's `.61` address.

Use `.lan`, not `.local`: `.local` is reserved for multicast DNS and can behave
differently across Linux, macOS, and mobile clients.

After the VMs exist and explicit approval is given, apply this atomic network
change:

1. select two unused VLAN 110 addresses and create fixed reservations for the
   `ai-control` and `ai-doris` VM MAC addresses;
2. verify all guests retain their addresses after DHCP renewal or restart;
3. create A records pointing the public service names to `ai-control`:
   `dstack.lan`, `trackio.lan`, `observatory.lan`, `registry.lan`,
   `beszel.lan`, and `doris.lan`;
4. create a private `doris-db.lan` A record pointing to `ai-doris`;
5. if a local object-store provider is later approved, reserve its address and
   add `s3.lan` and administrator-only `storage.lan` in that change;
6. route each current HTTP service by host name through Caddy or Traefik on
   `ai-control`;
7. issue an internally trusted TLS certificate for each name or a reviewed
   `*.lan` certificate;
8. verify resolution and TLS from the development machine and both workers.

`doris.lan` names the Doris web/API ingress only. Keep PostgreSQL, Doris's MySQL
protocol, Doris FE/BE internode ports, Docker sockets, and storage paths off the
reverse proxy and restrict them to the application network.

## Placement alternatives

| Rank | Placement | Strength | Cost or limit |
| ---: | --- | --- | --- |
| 1 | Dedicated `ai-control` and `ai-doris` Ubuntu VMs plus managed object storage | Separates database pressure from control services; no Dokploy dependency; fastest artifact path | Two guests to patch; managed storage requires approved network egress |
| 2 | One dedicated `ai-ops` Ubuntu VM | Fastest independent deployment; simple Ansible lifecycle | Doris failure or resource pressure can affect scheduler and tracking |
| 3 | Three small K3s VMs plus Doris Operator | Declarative reconciliation and the official Kubernetes Doris operator | Considerably more control-plane, storage, and networking work than two GPU workers need |
| 4 | Nomad clients in dedicated VMs | Good for a future mixed service/batch estate | Duplicates scheduling concepts while dstack remains the GPU-job scheduler |
| 5 | Direct Unraid containers | Fewest guests | Weakest isolation; Unraid has no native Compose support and the current 20 GB `docker.img` is unsuitable |
| 6 | Separate physical services server | Real failure-domain isolation and independent maintenance | Requires another reliable machine, storage, power, and backup path |

Select option 1. Provision:

- `ai-control`: Ubuntu LTS, 8–12 vCPU, 32–64 GiB RAM, and a 150–200 GB
  NVMe-backed OS/data disk;
- `ai-doris`: Ubuntu LTS, 16–24 vCPU, 96–128 GiB RAM, a small OS disk, and a
  separate 500–750 GB NVMe-backed Doris data disk;
- a constrained backup mount from each VM to the Unraid array;
- versioned Ansible inventory and roles that deploy pinned Compose control
  services and native Doris services.

Provision `ai-storage` only after the local-provider decision in
[`object-storage.md`](object-storage.md). It is not part of the first
deployment.

These are conservative starting allocations, not permanent reservations.
Measure actual CPU, memory, compaction, ingest, and disk growth during D2 and
reduce or expand them based on evidence. Do not run Doris directly inside
Unraid's loop-backed Docker image.

## Capacity and retention starting point

These are O1 trial bounds, not measured production capacity:

| Data | Hot location | Initial retention | Durable action |
| --- | --- | --- | --- |
| Doris metrics and compact events | dedicated NVMe disk | 30 days | Aggregate or expire by partition |
| Full traces | Doris only when queryable fields justify it | 7 days by default | Preserve selected traces as run artifacts |
| Trackio SQLite/Turso control metadata | dedicated NVMe volume | project lifetime | Engine-appropriate nightly backup |
| Artifact blobs | private Trackio-managed Hugging Face Storage Bucket initially | explicit per artifact | Preserve version, digest, aliases, and producer/consumer lineage in Trackio |
| Registry layers | registry volume on NVMe | bounded by referenced digests | Garbage-collect only after reference audit |
| Scheduler and service logs | separate bounded volumes | 7–14 days | Retain terminal summaries in execution evidence |

Every service requires a quota or retention rule before the first GPU run. A
low-disk alert at 70% and a scheduling stop at 80% protect the NVMe pool from
the unbounded run artifacts seen in earlier local experiments.

## O0 exit status

O0 is **conditional go**:

- compute, RAM, array capacity, NVMe capacity, and Docker availability are
  suitable;
- dedicated VMs avoid coupling this stack to Dokploy;
- all three service addresses must be reserved before DNS is added;
- VM creation, dedicated Doris data, and an off-host copy of Trackio artifacts remain
  release gates;
- single-host Doris is deliberately non-HA and must have a tested backup and
  restore path.
