# ai-stuff

AI projects on a **Jetson AGX Orin 64GB devkit** (`orin`), driven from a MacBook Air.
This root holds the box itself — setup, upgrade notes, shared monitoring; each project lives in
its own directory.

| Project | What |
|---|---|
| [`llm-serving/`](llm-serving/README.md) | Qwen3.6-35B-A3B on vLLM, Open WebUI chat on the Mac, LLM dashboards |

## Box setup (as of 2026-10-02)

| | |
|---|---|
| Hardware | Jetson AGX Orin 64GB devkit, 57 GB eMMC, 116 GB Lexar USB stick for models |
| OS | JetPack 7.2.1 (L4T R39.2.1), Ubuntu 24.04, kernel 6.8, driver 595.78 / CUDA 13.2 |
| Access | `ssh orin` (→ `orin.local`, LAN `end0`, 10.0.0.181), passwordless sudo, user `zx` |
| Power | `nvpmodel -m 0` (MAXN, persists) + `jetson_clocks` (re-applied by project start scripts) |
| Storage | model weights on `/mnt/models` (USB stick partition, ext4, `nofail` in fstab); docker images on eMMC (~13 GB free) |
| Headless console | micro-USB debug port → Mac `/dev/cu.usbmodemTOPO…E2` @115200 (UEFI + Linux console) |

## Layout

```
orin/                         # box-level, on the Jetson at ~/ai-stuff/orin
  jetson-exporter/            # Prometheus exporter for jtop (vendored, MIT) + EMC bandwidth patch
  monitoring/                 # docker compose: Prometheus + Grafana, shared by all projects
<project>/                    # e.g. llm-serving/ — its own README, orin/ and mac/ parts
```

Sync to the Jetson (Mac-only parts and local data stay behind):

```bash
rsync -a --exclude .git --exclude 'mac/' --exclude '.env' ~/ai-stuff/ orin:ai-stuff/
```

## Monitoring: Prometheus + Grafana

```bash
ssh orin 'cd ~/ai-stuff/orin/monitoring && sudo docker compose up -d'
open http://orin.local:3000                     # anonymous read-only; admin password:
ssh orin cat ~/ai-stuff/orin/monitoring/.env    # (not committed)
```

- Prometheus (:9090) scrapes the Jetson exporter (:9101) and project servers (`prometheus.yml`).
- Grafana (:3000) folders: **Orin box** — *Jetson Orin Exporter – jtop* from
  [pgodlews/jetson-orin-exporter](https://github.com/pgodlews/jetson-orin-exporter)
  ([grafana.com #25079](https://grafana.com/grafana/dashboards/25079-jetson-orin-exporter-jtop/));
  plus one folder per project, mounted from `<project>/orin/grafana-dashboards/` (see `docker-compose.yml`).

**Adding a project's metrics:** add a scrape job to `orin/monitoring/prometheus.yml`, put dashboards in
`<project>/orin/grafana-dashboards/`, then add a volume in `docker-compose.yml` and a provider in
`grafana/provisioning/dashboards/dashboards.yml`.

**Jetson exporter** — metrics from `jetson-stats` (jtop 7.2.2, `jtop.service`) via
`jetson_exporter.service` (port 9101, runs as root). Our patch (marked `[ai-stuff patch]`) adds
`jetson_emc_usage_percent` and `jetson_emc_bandwidth_bytes_per_second`
(= EMC clock × 2 (DDR) × 32 B (256-bit bus) × utilization) and samples every 2 s.
On JetPack 7 neither jtop (`EMC.val`) nor `tegrastats` reports EMC load any more, so the patch reads
the memory-controller activity monitor `/sys/kernel/debug/cactmon/mc_all` (root-only) and divides by
the EMC clock in kHz. NVML/DCGM-based GPU exporters do **not** work on Jetson.

## Upgrade notes (JetPack 5.1.2 → 7.2.1)

1. JP5 → JP7 needs a reflash; no apt path across major L4T versions (driver/CUDA are tied to the BSP).
2. The **Jetson ISO USB installer** works from a Mac (no x86 Linux host) but requires firmware ≥ r35.5:
   - r35.4.1 → r35.5 via apt: `sed -i s/r35.4/r35.5/ /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && apt dist-upgrade`
   - verify the capsule is staged (`OsIndications` set) **before** rebooting, and **after**:
     `sudo nvbootctrl dump-slots-info` → `Current version: 35.5.0`, `Capsule update status: 1`.
3. Write the ISO: `sudo dd if=jetsoninstaller-r39.2.1-arm64.iso of=/dev/rdiskN bs=4m status=progress`.
4. Boot it with the stick in a **USB-A** port (a stick in the device-mode USB-C port floods
   `tegra-xudc … PORTSC` errors and hangs networking). Watch the debug console; answer **Y** to the
   QSPI capsule prompt (30 s auto-skip!), pick *Install Jetson ISO* → *Install on eMMC (for AGX Orin)*
   (default highlight is NVMe).
5. First boot: oem-config over the USB-C serial (`/dev/cu.usbmodem…`). On JP7, Ethernet is `end0`,
   comes up DOWN until `ip link set end0 up && nmcli device connect end0`, and `openssh-server` must be enabled.
6. Docker on Ubuntu 24.04 lacks compose: `apt install docker-compose-v2`.
