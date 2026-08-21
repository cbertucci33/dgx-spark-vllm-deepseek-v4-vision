# Network configuration for two-node DGX Spark vision serving

This document intentionally contains no deployment-specific addresses, hostnames, usernames, interface names, or device identifiers. Discover and supply them on the target systems.

## Planes

| Plane | Purpose | Required value |
|---|---|---|
| RoCE fabric | NCCL tensor-parallel traffic and vLLM rendezvous | Dedicated fabric addresses and interfaces selected by the operator |
| Management / SSH | Deployment scripts and synchronization | Operator-managed SSH target or alias |
| API | OpenAI-compatible endpoint on the head node | Operator-selected bind address and port |

`MASTER_ADDR` and `VLLM_HOST_IP` must use the intended fabric plane unless the operator has explicitly validated another routed design.

## Discover target-specific values

Run these commands independently on both nodes:

```bash
ibdev2netdev
ip -br address
```

Record, but do not commit:

- head and worker fabric addresses;
- RoCE device identifiers;
- socket interface identifiers;
- the correct GID index;
- the worker SSH target;
- model, cache, temporary-storage, and repository paths.

Put those values only in ignored local `.env` files created from the publish-safe templates.

## Compose requirements

- `network_mode: host`
- `/dev/infiniband` device mount
- `NCCL_NET=IB`, `NCCL_IB_DISABLE=0`, `NCCL_IB_ROCE_VERSION_NUM=2`
- Explicit matching `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and `NCCL_IB_GID_INDEX` values on both ranks

## SSH from head to worker

The launch scripts use the configured worker SSH target. Configure passwordless key authentication through the target system's SSH configuration. Do not commit SSH aliases, usernames, addresses, ports, keys, or host fingerprints.

## Durability

Reverse proxies and boot-time service management are outside this recipe. The Compose stack does not automatically return after reboot unless the operator adds an external service manager.
